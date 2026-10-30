"""Short-budget FastGS with position learning-rate decay matched to the budget."""

if __package__:
    from .short_budget import main
else:
    from short_budget import main


if __name__ == "__main__":
    main(short_schedule=True)
