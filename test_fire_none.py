import fire

def train(
    resume_from_checkpoint: str = None,
):
    print(f"resume_from_checkpoint: {resume_from_checkpoint}")

if __name__ == "__main__":
    fire.Fire(train)
