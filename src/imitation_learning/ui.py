import tkinter as tk
from tkinter import messagebox
from typing import Optional, Callable, List, Dict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from card import Card, Suit, RANK_DISPLAY
from env import CuarentaEnv, Action, NUM_PLAYERS, TEAMS

# Suit icons
SUIT_ICONS = {
    Suit.BRILLOS: '♦️',           # Diamonds (Coins)
    Suit.CORAZONES_ROJOS: '♥️',   # Hearts (Cups)
    Suit.CORAZONES_NEGROS: '♠️',  # Spades (Swords)
    Suit.TREBOLES: '♣️',          # Clubs
}


class CuarentaUI:
    """Tkinter UI for Cuarenta game. Shows table and player hands, allows clicking cards."""

    def __init__(self, root: tk.Tk, env: CuarentaEnv):
        self.root = root
        self.env = env
        self.root.title("Cuarenta Imitation Learning UI")
        self.root.geometry("1200x800")

        self.selected_action: Optional[Action] = None
        self.waiting_for_action = False
        self.card_buttons: Dict[Card, tk.Button] = {}  # card -> button widget
        self.capture_buttons: Dict[int, tk.Button] = {}  # button id -> button
        self.pack_info: Dict[tk.Frame, Dict] = {}  # store pack parameters
        self.step_count = 0

        self._setup_ui()

    def _setup_ui(self):
        """Create main layout: center table, 4 player areas, info panel."""
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top section: Player 0 hand (dealer, bottom-right)
        self.p0_frame = tk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=2)
        self.p0_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        self.pack_info[self.p0_frame] = {"side": tk.TOP, "fill": tk.X, "pady": 5}
        tk.Label(self.p0_frame, text="P0 (Team 0)", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        self.p0_hand_frame = tk.Frame(self.p0_frame)
        self.p0_hand_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Middle section: table + left/right players
        middle_frame = tk.Frame(main_frame)
        middle_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=5)

        # Left player (Player 3, team 1)
        self.p3_frame = tk.Frame(middle_frame, relief=tk.SUNKEN, borderwidth=2, width=100)
        self.p3_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5)
        self.pack_info[self.p3_frame] = {"side": tk.LEFT, "fill": tk.BOTH, "padx": 5}
        tk.Label(self.p3_frame, text="P3\n(Team 1)", font=("Arial", 12, "bold")).pack(pady=5)
        self.p3_hand_frame = tk.Frame(self.p3_frame)
        self.p3_hand_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

        # Center table
        self.table_frame = tk.Frame(middle_frame, bg="gray50", relief=tk.SUNKEN, borderwidth=3)
        self.table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        self.table_label = tk.Label(self.table_frame, text="Table (empty)", bg="gray50", fg="white", font=("Arial", 10))
        self.table_label.pack(padx=20, pady=20, expand=True)

        self.table_cards_frame = tk.Frame(self.table_frame, bg="gray50")
        self.table_cards_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Right player (Player 1, team 1)
        self.p1_frame = tk.Frame(middle_frame, relief=tk.SUNKEN, borderwidth=2, width=100)
        self.p1_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)
        self.pack_info[self.p1_frame] = {"side": tk.RIGHT, "fill": tk.BOTH, "padx": 5}
        tk.Label(self.p1_frame, text="P1\n(Team 1)", font=("Arial", 12, "bold")).pack(pady=5)
        self.p1_hand_frame = tk.Frame(self.p1_frame)
        self.p1_hand_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)

        # Bottom section: Player 2 hand (partner of P0)
        self.p2_frame = tk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=2)
        self.p2_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        self.pack_info[self.p2_frame] = {"side": tk.BOTTOM, "fill": tk.X, "pady": 5}
        tk.Label(self.p2_frame, text="P2 (Team 0)", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        self.p2_hand_frame = tk.Frame(self.p2_frame)
        self.p2_hand_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Info panel
        info_frame = tk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=1)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        self.info_label = tk.Label(info_frame, text="", font=("Arial", 9), justify=tk.LEFT)
        self.info_label.pack(padx=5, pady=2, fill=tk.X)

        # Action info
        self.action_label = tk.Label(info_frame, text="", font=("Arial", 9, "bold"), fg="white")
        self.action_label.pack(padx=5, pady=2, fill=tk.X)

    def _card_to_label(self, card: Card) -> str:
        """Convert card to display string with rank + suit icon."""
        return f"{RANK_DISPLAY[card.rank]}{SUIT_ICONS[card.suit]}"

    def _make_card_button(self, card: Card, callback: Callable) -> tk.Button:
        """Create a card button (white box with rank+suit icon)."""
        btn = tk.Button(
            text=self._card_to_label(card),
            width=6,
            height=3,
            bg="white",
            fg="black",
            font=("Arial", 16, "bold"),
            relief=tk.RAISED,
            borderwidth=3,
            command=callback
        )
        return btn

    def refresh_display(self):
        """Update entire display based on current env state."""
        s = self.env.state

        # Clear old card buttons
        for btn in self.card_buttons.values():
            btn.pack_forget()
        self.card_buttons.clear()

        for frame in [self.p0_hand_frame, self.p1_hand_frame, self.p2_hand_frame, self.p3_hand_frame]:
            for w in frame.winfo_children():
                w.destroy()

        # Update each player's hand
        player_frames = [self.p0_hand_frame, self.p1_hand_frame, self.p2_hand_frame, self.p3_hand_frame]
        player_parent_frames = [self.p0_frame, self.p1_frame, self.p2_frame, self.p3_frame]
        current_player = s.current_player

        for player_id in range(NUM_PLAYERS):
            hand = s.hands[player_id]
            frame = player_frames[player_id]
            parent_frame = player_parent_frames[player_id]

            if player_id == current_player:
                # Show current player's hand
                parent_frame.config(bg="yellow", relief=tk.RIDGE, borderwidth=4)

                for card in sorted(hand):
                    def make_callback(c, pid):
                        return lambda: self._on_card_clicked(c, pid)

                    btn = self._make_card_button(card, make_callback(card, player_id))
                    btn.pack(side=tk.LEFT, padx=2, pady=2)
                    self.card_buttons[card] = btn
            else:
                # Hide other players' hands by keeping frame visible but empty
                parent_frame.config(bg="SystemButtonFace", relief=tk.SUNKEN, borderwidth=2)

        # Update table
        self._update_table_display()

        # Update info
        team0_score = s.scores[0]
        team1_score = s.scores[1]
        team0_captured = len(s.cards_captured[0])
        team1_captured = len(s.cards_captured[1])
        current_team = TEAMS[current_player]

        info_text = f"Step: {self.step_count} | Team 0 (P0,P2): {team0_score} pts, {team0_captured} cards | Team 1 (P1,P3): {team1_score} pts, {team1_captured} cards | Current: Player {current_player} (Team {current_team})"
        self.info_label.config(text=info_text)

    def _update_table_display(self):
        """Update table display with current cards."""
        for w in self.table_cards_frame.winfo_children():
            w.destroy()

        s = self.env.state
        table = s.table

        if not table:
            self.table_label.config(text="Table (empty)")
        else:
            self.table_label.config(text=f"Table ({len(table)} cards)")

            cards_row = tk.Frame(self.table_cards_frame, bg="gray50")
            cards_row.pack(anchor=tk.CENTER, expand=True, padx=5, pady=5)

            for i, card in enumerate(table):
                is_last = (i == len(table) - 1)
                card_bg = "#90EE90" if is_last else "white"  # Light green for last card
                card_label = tk.Label(
                    cards_row,
                    text=self._card_to_label(card),
                    width=6,
                    height=3,
                    bg=card_bg,
                    fg="black",
                    font=("Arial", 16, "bold"),
                    relief=tk.RAISED,
                    borderwidth=3
                )
                card_label.pack(side=tk.LEFT, padx=2, pady=2)

    def _on_card_clicked(self, card: Card, player_id: int):
        """Handle card click. Show legal actions for this card."""
        s = self.env.state

        if s.current_player != player_id:
            messagebox.showwarning("Not your turn", f"It's player {s.current_player}'s turn")
            return

        # Get all legal actions for this card
        legal_actions = [a for a in self.env.legal_actions() if a.card == card]

        if not legal_actions:
            messagebox.showwarning("No legal actions", f"Card {card} cannot be played")
            return

        if len(legal_actions) == 1:
            # Only one option, play it
            self.selected_action = legal_actions[0]
            self._confirm_action()
        else:
            # Multiple capture options, show selection dialog
            self._show_capture_options(card, legal_actions)

    def _show_capture_options(self, card: Card, actions: List[Action]):
        """Show popup to select which capture to perform (or just place)."""
        popup = tk.Toplevel(self.root)
        popup.title(f"Choose action for {card}")
        popup.geometry("300x200")

        tk.Label(popup, text=f"Played: {card}", font=("Arial", 10, "bold")).pack(pady=5)
        tk.Label(popup, text="Select capture option:", font=("Arial", 9)).pack(pady=5)

        def make_callback(action):
            return lambda: self._select_action(action, popup)

        for action in actions:
            if action.capture:
                cap_text = " + ".join(self._card_to_label(c) for c in sorted(action.capture))
                btn_text = f"Capture: {cap_text}"
            else:
                btn_text = "Just place (no capture)"

            btn = tk.Button(popup, text=btn_text, command=make_callback(action), width=35)
            btn.pack(pady=2, padx=5, fill=tk.X)

    def _select_action(self, action: Action, popup: tk.Toplevel):
        """User selected an action."""
        self.selected_action = action
        popup.destroy()
        self._confirm_action()

    def _confirm_action(self):
        """Confirm the selected action. Trainer will step the environment."""
        if not self.selected_action:
            return

        action = self.selected_action
        self.action_label.config(text=f"Selected: {action}")
        self.step_count += 1

    def wait_for_action(self) -> Action:
        """Block until player selects and confirms an action. Return the action."""
        self.selected_action = None

        # Refresh display
        self.refresh_display()

        # Wait for action selection
        while self.selected_action is None:
            self.root.update()

        return self.selected_action


def run_ui_demo():
    """Run interactive UI demo for data collection."""
    root = tk.Tk()
    env = CuarentaEnv()
    env.reset()

    ui = CuarentaUI(root, env)
    ui.refresh_display()

    root.mainloop()


if __name__ == "__main__":
    run_ui_demo()
