from _40 import *
from tkinter import *
from tkinter import ttk, messagebox

###
# Game process
# 1. Each player receives 5 cards.
# 2. Player plays its card (caida, suma, escalera, or just throwing some card without explicit secondary effect)
# possibly picking cards from the table and/or receiving tantos from the judge.
# 3. Play the partial until all players have run out of cards or some team has won (reached 40 by caida).
# 4. If the partial ended and there's still deck cards, give 5 cards to each player again and repeat the other steps.
# 5. If the partial ended and there's no deck cards, the starting turn moves one person to the right.
# Give 5 cards to each player and repeat the other steps.
###


def translate_card_type(type):
    types = {
        "clubs": "Tréboles",
        "hearts": "Corazones",
        "diamonds": "Diamantes",
        "spades": "Espadas",
    }

    return types[type]


def convert_card_set_to_list(val):
    return list(map(lambda x: f"{x[0]} - {translate_card_type(x[1])}", list(val)))


if __name__ == "__main__":
    game = Game()

    root = Tk()
    selected_value = StringVar(value="A")
    selected_action = IntVar(value=0)
    table_cards = StringVar(value=convert_card_set_to_list(game.table))
    ai_hand_cards = StringVar(value=convert_card_set_to_list(game.ai_hand))
    main_controls = ttk.Frame(root, padding=10)
    main_controls.grid()

    def handle_reset_click(type):
        def callback():
            if type == "new_round":
                game.new_round()
            if type == "new_game":
                game.new_game()

            re_render_dynamic_widgets()

        return callback

    ttk.Button(main_controls, text="AI", command=game.play_ai_turn).grid(
        column=0, row=0
    )
    ttk.Button(
        main_controls, text="Nueva ronda", command=handle_reset_click("new_round")
    ).grid(column=1, row=0)
    ttk.Button(
        main_controls, text="Nuevo juego", command=handle_reset_click("new_game")
    ).grid(column=2, row=0)

    ttk.Label(main_controls, text="MESA").grid(column=3, row=0)
    Listbox(
        main_controls, listvariable=table_cards, height=15, width=50, borderwidth=2
    ).grid(column=3, row=1)

    ttk.Label(main_controls, text="MANO AI").grid(column=4, row=0)
    Listbox(
        main_controls, listvariable=ai_hand_cards, height=15, width=50, borderwidth=2
    ).grid(column=4, row=1)

    card_registry_controls = ttk.Frame(root, padding=10)
    card_registry_controls.grid()

    def re_render_dynamic_widgets():
        table_cards.set(convert_card_set_to_list(game.table))
        ai_hand_cards.set(value=convert_card_set_to_list(game.ai_hand))

    def handle_card_type_click(card_type):
        def callback():
            card_value = selected_value.get()
            action = selected_action.get()

            if card_value not in card_values:
                messagebox.showinfo(
                    message=f"'{card_value}' inválido", title="Shunsho!"
                )
                return

            if action == 0:
                game.put_card_on_table((card_value, card_type))
                game.exec_turn((card_value, card_type))
            if action == 1:
                game.put_card_in_ai_hand((card_value, card_type))
            if action == 2:
                game.remove_card_from_table((card_value, card_type))
            if action == 3:
                game.remove_card_from_ai_hand((card_value, card_type))

            re_render_dynamic_widgets()

        return callback

    ttk.Label(card_registry_controls, text="Nueva carta").grid(column=0, row=0)
    ttk.Entry(card_registry_controls, textvariable=selected_value).grid(column=0, row=1)

    ttk.Button(
        card_registry_controls, text="Tréboles", command=handle_card_type_click("clubs")
    ).grid(column=0, row=2)
    ttk.Button(
        card_registry_controls,
        text="Corazones",
        command=handle_card_type_click("hearts"),
    ).grid(column=0, row=3)
    ttk.Button(
        card_registry_controls,
        text="Diamantes",
        command=handle_card_type_click("diamonds"),
    ).grid(column=0, row=4)
    ttk.Button(
        card_registry_controls, text="Espadas", command=handle_card_type_click("spades")
    ).grid(column=0, row=5)

    ttk.Label(card_registry_controls, text="Acción").grid(column=0, row=6)

    ttk.Radiobutton(
        card_registry_controls,
        text="Poner carta en mesa",
        variable=selected_action,
        value=0,
    ).grid(column=0, row=7)

    ttk.Radiobutton(
        card_registry_controls,
        text="Poner carta en mano",
        variable=selected_action,
        value=1,
    ).grid(column=0, row=8)

    ttk.Radiobutton(
        card_registry_controls,
        text="Quita carta de mesa",
        variable=selected_action,
        value=2,
    ).grid(column=0, row=9)

    ttk.Radiobutton(
        card_registry_controls,
        text="Quitar carta de mano",
        variable=selected_action,
        value=3,
    ).grid(column=0, row=10)

    root.mainloop()
