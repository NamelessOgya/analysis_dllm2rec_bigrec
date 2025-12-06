
from tqdm import tqdm

def main():
    instructions = list(range(100))
    inputs = list(range(100))
    batch_size = 10

    def batch(list, batch_size=batch_size):
        chunk_size = (len(list) - 1) // batch_size + 1
        for i in range(chunk_size):
            yield list[batch_size * i: batch_size * (i + 1)]

    print("DEBUG: Starting inference loop...")
    total_batches = (len(instructions) - 1) // batch_size + 1
    
    # Mimic the shadowing
    for i, batch in tqdm(enumerate(zip(batch(instructions), batch(inputs))), total=total_batches):
        print(f"DEBUG: Processing batch {i}...")
        inst, inp = batch
        print(f"Loop var batch type: {type(batch)}")
    
    print("Done")

if __name__ == "__main__":
    main()
