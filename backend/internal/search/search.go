package search

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"cua/internal/game"
)

// Value target names, matching `src/ismcts/network.py: VALUE_TARGETS` and the
// `value_target` field of the ONNX sidecar written by the exporter.
const (
	ValueTargetWin    = "win"
	ValueTargetReturn = "return"
)

// Config is one search's parameters.
type Config struct {
	// Iterations is the number of simulations per decision (the Python side's `n_iter`).
	// This is the strength/latency knob: MEASURED on `checkpoints-return/best_heuristic.pt`
	// against the heuristic, per AI turn: 0 sims (policy only) 49.0%, 32 sims 57.0%,
	// 64 sims 59.0%, 128 sims 72.0%, 400 sims 77.0%.
	Iterations int

	// CPuct is PUCT's exploration constant. 1.5 is the value the Python search uses.
	CPuct float64

	// DirichletEps is root exploration noise, mixed into the priors BEFORE the search.
	// It belongs to self-play (it is how the training targets stay exploratory); playing a
	// game should leave it at 0, exactly like the Python evaluation harnesses.
	DirichletEps   float64
	DirichletAlpha float64

	// ValueTarget is the mode the weights were trained in: "return" makes the search
	// accumulate the dense reward and read a linear head, "win" backs up ±1 results and
	// ignores rewards. It MUST match the model, or the search optimises a different
	// objective than the one the policy learned (and the head's output is on the wrong
	// scale). It comes from the same sidecar the exporter wrote.
	ValueTarget string

	// ValueScale is RETURN_VALUE_SCALE: the divisor between game points and the head's
	// units in return mode.
	ValueScale float64
}

// DefaultConfig is a reasonable server-side default once the budget is chosen by the
// caller; `Iterations` is left at 0 so a zero value never silently searches.
func DefaultConfig() Config {
	return Config{
		Iterations:     0,
		CPuct:          1.5,
		DirichletEps:   0.0,
		DirichletAlpha: 0.3,
		ValueTarget:    ValueTargetWin,
		ValueScale:     1.0,
	}
}

// HistoryStep is one completed decision of the seat the search is playing for: the
// observation it saw and the action-features of the move it made. The model's root
// evaluation reads this window (its interior nodes see a single pending step).
type HistoryStep struct {
	Obs   []float32
	Feats []float32
}

// Evaluator runs one model call. The search owns the game logic and the tensor layout, so
// an implementation only has to feed the ONNX session — which is what makes the search
// testable with a fake evaluator (see search_test.go).
type Evaluator interface {
	// Forward returns the policy logits (one per slot, illegal slots already masked by the
	// model) and the value for the player to move.
	//
	//	obsSeq   [steps][208]   the observation sequence, LAST step = the pending decision
	//	actSeq   [steps][84]    action-features of the actions taken at each completed step
	//	actMask  [steps]         1 where a history step is real, 0 for padding/pending
	//	feats    [64][84]       the candidate actions of THIS decision, legal ones first
	//	mask     [64]           1 on the legal slots
	Forward(obsSeq, actSeq, actMask, feats, mask []float32, steps int) ([]float32, float32, error)

	// Steps is the observation window the loaded model declares (1 = single pending step,
	// i.e. no history; see scripts/pth2onnx_ismcts.py --window).
	Steps() int
}

// ModelEvaluator is an Evaluator that also knows the model's settings — the value target
// and its scale — so the server can build a Config without knowing which model is loaded.
// Keeping this an interface is what lets the backend's AI-turn wiring be tested with a stub
// instead of a real ONNX session.
type ModelEvaluator interface {
	Evaluator
	Config(iterations int, cpuct float64) Config
}

// Search is one decision's worth of ISMCTS. Build it per decision (a fresh tree, no reuse
// across plies — the same choice the Python search makes, because determinizations do not
// compose across moves).
type Search struct {
	cfg      Config
	eval     Evaluator
	rng      *rand.Rand
	rootSeat int

	stats Stats
}

// Stats is the diagnostics one decision produced.
type Stats struct {
	Simulations  int
	Evaluations  int
	RootValue    float64 // root team's POV, in the model's units
	RootValuePts float64 // the same, in game points (return mode)
	Iterations   int

	// Actions and Visits are the root's legal actions and their visit counts, aligned:
	// Visits[i] is how many simulations chose Actions[i]. That is the policy target the
	// Python search would train on, and it is what a caller should log to see how
	// concentrated (or conflicted) the search's opinion was.
	Actions []game.Action
	Visits  []int
}

// New builds a search over `state` for `rootSeat`, which must be the seat to move.
func New(state *game.State, rootSeat int, eval Evaluator, cfg Config, rng *rand.Rand) (*Search, error) {
	if eval == nil {
		return nil, fmt.Errorf("no evaluator")
	}
	if state == nil || state.Done {
		return nil, fmt.Errorf("no live position to search")
	}
	if state.CurrentPlayer != rootSeat {
		return nil, fmt.Errorf("seat %d is to move, not %d", state.CurrentPlayer, rootSeat)
	}
	if cfg.Iterations <= 0 {
		return nil, fmt.Errorf("search needs Iterations > 0")
	}
	if cfg.CPuct <= 0 {
		cfg.CPuct = 1.5
	}
	if cfg.ValueTarget == "" {
		cfg.ValueTarget = ValueTargetWin
	}
	if cfg.ValueTarget == ValueTargetReturn && cfg.ValueScale <= 0 {
		return nil, fmt.Errorf("return mode needs ValueScale (RETURN_VALUE_SCALE)")
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}
	return &Search{cfg: cfg, eval: eval, rng: rng, rootSeat: rootSeat}, nil
}

// RandomizePriors is how the tests and any callers that want reproducible-yet-varied
// determinizations set the RNG (the search already holds it).
func (s *Search) RandomizePriors() {}

// Decide runs the simulations and returns the action to play (the most visited one, ties
// broken by the priors) plus this decision's statistics.
//
// `history` is the root seat's own past decisions, oldest first; it is only used when the
// model declares a window longer than one step.
func (s *Search) Decide(state *game.State, history []HistoryStep) (game.Action, Stats, error) {
	rootTeam := s.rootSeat % 2
	legal := LegalActions(state)
	if len(legal) == 0 {
		return game.Action{}, s.stats, fmt.Errorf("no legal actions")
	}
	if len(legal) > game.MaxActionSlots {
		return game.Action{}, s.stats, fmt.Errorf("%d legal actions but only %d policy slots",
			len(legal), game.MaxActionSlots)
	}

	s.stats.Actions = nil
	s.stats.Visits = nil
	root := newNode(s.rootSeat)
	priors, rootValue, err := s.evaluateRoot(state, legal, history)
	if err != nil {
		return game.Action{}, s.stats, err
	}
	s.stats.Evaluations++
	s.stats.RootValue = rootValue
	s.stats.RootValuePts = rootValue
	if s.cfg.ValueTarget == ValueTargetReturn {
		s.stats.RootValuePts = rootValue * s.cfg.ValueScale
	}

	if s.cfg.DirichletEps > 0 && len(legal) > 1 {
		noise := dirichlet(s.rng, s.cfg.DirichletAlpha, len(legal))
		for i := range priors {
			priors[i] = (1-s.cfg.DirichletEps)*priors[i] + s.cfg.DirichletEps*noise[i]
		}
	}

	root.evaluated = true
	root.priors = priors
	root.priorActions = legal
	root.valueMoverPOV = rootValue
	for i, action := range legal {
		child := newNode(-1)
		child.prior = priors[i]
		root.children[actionKey(action)] = child
		root.order = append(root.order, action)
	}

	for i := 0; i < s.cfg.Iterations; i++ {
		world, err := Sample(s.rootSeat, state, s.rng)
		if err != nil {
			return game.Action{}, s.stats, err
		}
		if err := s.simulate(root, world, rootTeam); err != nil {
			return game.Action{}, s.stats, err
		}
	}

	s.stats.Simulations = s.cfg.Iterations
	s.stats.Iterations = s.cfg.Iterations

	best := root.order[0]
	bestVisits := -1
	for _, action := range root.order {
		child := root.children[actionKey(action)]
		s.stats.Actions = append(s.stats.Actions, action)
		s.stats.Visits = append(s.stats.Visits, child.n)
		if child.n > bestVisits {
			best, bestVisits = action, child.n
		}
	}
	return best, s.stats, nil
}

// Stats returns the stats of the last Decide.
func (s *Search) Stats() Stats { return s.stats }

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------

type node struct {
	n             int
	w             float64 // sum of values in ROOT-team POV
	children      map[string]*node
	order         []game.Action // the actions this node was expanded with
	priorActions  []game.Action
	priors        []float64
	player        int
	team          int
	prior         float64
	valueMoverPOV float64
	evaluated     bool
}

func newNode(player int) *node {
	team := -1
	if player >= 0 {
		team = player % 2
	}
	return &node{children: map[string]*node{}, player: player, team: team}
}

// priorFor returns the network's prior for `action` at this node. A later determinization
// can make an action legal that was illegal when the node was evaluated (the mover's hand
// is sampled), and those actions have no prior; they get the mean of the known ones so they
// are neither favoured nor starved.
func (n *node) priorFor(action game.Action) float64 {
	if len(n.priors) == 0 {
		return 1.0
	}
	for i, candidate := range n.priorActions {
		if sameAction(candidate, action) {
			return n.priors[i]
		}
	}
	sum := 0.0
	for _, p := range n.priors {
		sum += p
	}
	return sum / float64(len(n.priors))
}

// sameAction compares by card and capture set (order-insensitive).
func sameAction(a, b game.Action) bool {
	return a.Card == b.Card && sameCapture(a.Capture, b.Capture)
}

func actionKey(action game.Action) string {
	key := fmt.Sprintf("%d|", action.Card.Index())
	cards := canonicalCards(action.Capture)
	for i, card := range cards {
		if i > 0 {
			key += ","
		}
		key += fmt.Sprintf("%d", card.Index())
	}
	return key
}

// ---------------------------------------------------------------------------
// Simulation
// ---------------------------------------------------------------------------

func (s *Search) simulate(root *node, state *game.State, rootTeam int) error {
	cur := root
	type edge struct {
		parent *node
		action game.Action
	}
	path := make([]edge, 0, 16)

	var valueRootPOV float64
	collected := 0.0 // return mode: rewards gathered on the way down, root POV

	for {
		if state.Done {
			valueRootPOV = s.terminalValue(state, rootTeam)
			break
		}

		legal := LegalActions(state)
		if len(legal) == 0 {
			valueRootPOV = 0 // unreachable in cuarenta; keep the simulation alive
			break
		}

		missing := make([]game.Action, 0, len(legal))
		for _, action := range legal {
			if _, ok := cur.children[actionKey(action)]; !ok {
				missing = append(missing, action)
			}
		}

		if len(missing) > 0 {
			if !cur.evaluated {
				priors, value, err := s.evaluateNode(state, legal)
				if err != nil {
					return err
				}
				cur.evaluated = true
				cur.priors = priors
				cur.priorActions = legal
				cur.valueMoverPOV = value
			}
			for _, action := range missing {
				child := newNode(-1)
				child.prior = cur.priorFor(action)
				cur.children[actionKey(action)] = child
				cur.order = append(cur.order, action)
			}
			valueRootPOV = s.toRootPOV(cur.valueMoverPOV, cur.team, rootTeam)
			break
		}

		action, child, err := s.selectChild(cur, legal, rootTeam)
		if err != nil {
			return err
		}
		actorTeam := state.CurrentPlayer % 2
		next, result, err := StepState(state, action, s.rng)
		if err != nil {
			return err
		}
		if s.cfg.ValueTarget == ValueTargetReturn {
			collected += s.shapedValue(result.Reward, actorTeam, rootTeam)
		}
		path = append(path, edge{parent: cur, action: action})
		state = next
		cur = child
		cur.player = state.CurrentPlayer
		cur.team = state.CurrentPlayer % 2
	}

	backup := valueRootPOV + collected
	for _, e := range path {
		child := e.parent.children[actionKey(e.action)]
		child.n++
		child.w += backup
	}
	root.n++

	return nil
}

// terminalValue is the value of a finished game in the model's units, root POV.
//
//	return -> 0: no rewards remain (the reward of the winning ply is on that edge, so the
//	          simulation's value still ends up equal to the game's shaped return)
//	win    -> ±1, the game's zero-sum payoff
func (s *Search) terminalValue(state *game.State, rootTeam int) float64 {
	if s.cfg.ValueTarget == ValueTargetReturn {
		return 0
	}
	if state.Winner != nil && *state.Winner == rootTeam {
		return 1
	}
	return -1
}

// shapedValue converts one step's reward into a contribution to the root team's value:
// positive when the root's team was paid, negative when the opponent was, in the model's
// units.
//
// This mirror of `src/ppo/rollout.py: shaped_reward` is a SHAPING choice, not a fact about
// the game (the env's reward is not zero-sum — both teams are paid for captures). It is the
// same one the Python search uses, because a single scalar Q is what PUCT compares.
func (s *Search) shapedValue(reward float64, actorTeam, rootTeam int) float64 {
	sign := 1.0
	if actorTeam != rootTeam {
		sign = -1.0
	}
	return sign * reward / s.cfg.ValueScale
}

func (s *Search) toRootPOV(valueMoverPOV float64, moverTeam, rootTeam int) float64 {
	if moverTeam < 0 || moverTeam == rootTeam {
		return valueMoverPOV
	}
	return -valueMoverPOV
}

// selectChild is PUCT: U(s,a) = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a)).
//
// Q is expressed in the MOVER's POV (converted from root POV), so the mover maximises it.
// Unvisited children are NEUTRAL: in win mode that is Q = 0 (a value symmetric around 0
// makes "even" the natural unknown), and in return mode it is the node's OWN value, because
// a return is large and positive — scoring a fresh child 0 there would look catastrophic
// and the search would never try it.
func (s *Search) selectChild(cur *node, legal []game.Action,
	rootTeam int) (game.Action, *node, error) {
	if cur.team < 0 {
		return game.Action{}, nil, fmt.Errorf("PUCT on a node whose mover is unknown")
	}
	sqrtN := math.Sqrt(math.Max(float64(cur.n), 1))

	center := 0.0
	if s.cfg.ValueTarget == ValueTargetReturn {
		center = cur.valueMoverPOV
	}

	bestScore := math.Inf(-1)
	var bestAction game.Action
	var bestChild *node

	for _, action := range legal {
		child, ok := cur.children[actionKey(action)]
		if !ok {
			continue
		}
		var q float64
		if child.n == 0 {
			q = center
		} else {
			q = s.toRootPOV(child.w/float64(child.n), cur.team, rootTeam)
		}
		score := (q - center) + s.cfg.CPuct*child.prior*sqrtN/(1+float64(child.n))
		if score > bestScore {
			bestScore, bestAction, bestChild = score, action, child
		}
	}
	if bestChild == nil {
		return game.Action{}, nil, fmt.Errorf("PUCT found no legal child")
	}
	return bestAction, bestChild, nil
}

// ---------------------------------------------------------------------------
// Network plumbing
// ---------------------------------------------------------------------------

func (s *Search) evaluateRoot(state *game.State, legal []game.Action,
	history []HistoryStep) ([]float64, float64, error) {
	steps := s.eval.Steps()
	if steps < 1 {
		steps = 1
	}
	obs := game.FlattenObservation(game.ObserveState(state, s.rootSeat))
	feats, mask, slots := game.BuildActionFeatures(legal, state)
	if slots != len(legal) {
		return nil, 0, fmt.Errorf("%d legal actions but only %d slots", len(legal), slots)
	}

	obsSeq := make([]float32, 0, steps*game.ObsSizeISMCTS)
	actSeq := make([]float32, 0, steps*game.ActionFeatureDim)
	actMask := make([]float32, steps)

	// Left padding, matching the training dataset: the pending step must be LAST, and any
	// history shorter than the window is padded at the front.
	pad := steps - 1 - min(len(history), steps-1)
	for i := 0; i < pad; i++ {
		obsSeq = append(obsSeq, make([]float32, game.ObsSizeISMCTS)...)
		actSeq = append(actSeq, make([]float32, game.ActionFeatureDim)...)
	}
	keep := min(len(history), steps-1)
	for _, step := range history[len(history)-keep:] {
		obsSeq = append(obsSeq, step.Obs...)
		actSeq = append(actSeq, step.Feats...)
	}
	for i := 0; i < keep; i++ {
		actMask[pad+i] = 1
	}
	obsSeq = append(obsSeq, obs...)
	actSeq = append(actSeq, make([]float32, game.ActionFeatureDim)...)

	logits, value, err := s.eval.Forward(obsSeq, actSeq, actMask, feats.Flatten(),
		mask.Flatten(), steps)
	if err != nil {
		return nil, 0, err
	}
	return softmaxPrefix(logits, len(legal)), float64(value), nil
}

func (s *Search) evaluateNode(state *game.State, legal []game.Action) ([]float64, float64, error) {
	mover := state.CurrentPlayer
	obs := game.FlattenObservation(game.ObserveState(state, mover))
	feats, mask, slots := game.BuildActionFeatures(legal, state)
	if slots != len(legal) {
		return nil, 0, fmt.Errorf("%d legal actions but only %d slots", len(legal), slots)
	}
	actSeq := make([]float32, game.ActionFeatureDim)
	actMask := []float32{0}
	logits, value, err := s.eval.Forward(obs, actSeq, actMask, feats.Flatten(),
		mask.Flatten(), 1)
	if err != nil {
		return nil, 0, err
	}
	s.stats.Evaluations++
	return softmaxPrefix(logits, len(legal)), float64(value), nil
}

// softmaxPrefix is the softmax over the first `n` logits (the legal slots — the model has
// already written -1e9 on the illegal ones, and the caller packed legal actions first).
func softmaxPrefix(logits []float32, n int) []float64 {
	if n > len(logits) {
		n = len(logits)
	}
	out := make([]float64, n)
	if n == 0 {
		return out
	}
	maxLogit := float64(logits[0])
	for i := 1; i < n; i++ {
		if float64(logits[i]) > maxLogit {
			maxLogit = float64(logits[i])
		}
	}
	sum := 0.0
	for i := 0; i < n; i++ {
		out[i] = math.Exp(float64(logits[i]) - maxLogit)
		sum += out[i]
	}
	if sum <= 0 {
		for i := range out {
			out[i] = 1 / float64(n)
		}
		return out
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// dirichlet draws a symmetric Dirichlet sample (the root exploration noise).
func dirichlet(rng *rand.Rand, alpha float64, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = gammaSample(rng, alpha)
	}
	sum := 0.0
	for _, v := range out {
		sum += v
	}
	if sum <= 0 {
		for i := range out {
			out[i] = 1 / float64(n)
		}
		return out
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// gammaSample is Marsaglia-Tsang; AlphaZero's noise only needs a plausible Dirichlet.
func gammaSample(rng *rand.Rand, alpha float64) float64 {
	if alpha <= 0 {
		return 1
	}
	if alpha < 1 {
		// Boost: Gamma(a) = Gamma(a+1) * U^(1/a)
		u := rng.Float64()
		for u == 0 {
			u = rng.Float64()
		}
		return gammaSample(rng, alpha+1) * math.Pow(u, 1/alpha)
	}
	d := alpha - 1.0/3.0
	c := 1.0 / math.Sqrt(9*d)
	for {
		x := rng.NormFloat64()
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := rng.Float64()
		if u < 1-0.0331*x*x*x*x {
			return d * v
		}
		if math.Log(u) < 0.5*x*x+d*(1-v+math.Log(v)) {
			return d * v
		}
	}
}

// SortedVisits is a diagnostic: the visited actions, most visited first.
func SortedVisits(root *node) []game.Action {
	actions := append([]game.Action(nil), root.order...)
	sort.SliceStable(actions, func(i, j int) bool {
		return root.children[actionKey(actions[i])].n > root.children[actionKey(actions[j])].n
	})
	return actions
}
