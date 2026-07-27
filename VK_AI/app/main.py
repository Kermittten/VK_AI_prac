from llm.model import load_model
from llm.inference import generate_answer


def main():

    model, processor = load_model()

    result, raw_answer = generate_answer(
        model,
        processor,
        "tests/test.jpg"
    )

    print(result)

    print()

    print(raw_answer)


if __name__ == "__main__":
    main()