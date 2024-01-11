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

if __name__ == "__main__":
    game = Game()

    root = Tk()
    selected_value = StringVar(value="A")
    selected_action = IntVar(value=0)
    card_registry_controls = ttk.Frame(root, padding=10)
    card_registry_controls.grid()

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
            if action == 1:
                game.put_card_in_ai_hand((card_value, card_type))

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

    root.mainloop()
