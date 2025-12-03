import fire

def train(
    train_on_inputs: bool = True,
    group_by_length: bool = False,
):
    print(f"train_on_inputs: {train_on_inputs} (type: {type(train_on_inputs)})")
    print(f"group_by_length: {group_by_length} (type: {type(group_by_length)})")

if __name__ == "__main__":
    fire.Fire(train)
