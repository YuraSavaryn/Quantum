import argparse
from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Mountain NER Model")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./model",
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Last summer, my friends and I traveled to Nepal to see Mount Everest, but next year we hope to tackle K2 or Mount Kilimanjaro.",
        help="Text to analyze for mountain names"
    )
    return parser.parse_args()


def inference_ner():
    args = parse_args()

    print(f"Loading model and tokenizer from: {args.model_dir}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    except Exception as e:
        print(f"Error loading model from '{args.model_dir}': {e}")
        tokenizer = AutoTokenizer.from_pretrained("Sava777/mountain_ner")
        model = AutoModelForTokenClassification.from_pretrained("Sava777/mountain_ner")

    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple"
    )

    print(f"\nInput Text: \n> \"{args.text}\"\n")
    results = ner_pipeline(args.text)

    if not results:
        print("Result: No mountain names detected in the text.")
        return

    grouped_entities = []
    current_entity = None

    for entity in results:
        label = entity.get("entity_group", entity.get("entity"))

        if label == "O" or not any(target in label for target in ["MOUNTAIN"]):
            continue

        start = entity.get("start")
        end = entity.get("end")
        score = entity.get("score")

        if start is None or end is None:
            continue

        if current_entity is None:
            current_entity = {"start": start, "end": end, "scores": [score]}
        else:
            gap = args.text[current_entity["end"]:start]
            if gap.strip() == "":
                current_entity["end"] = end
                current_entity["scores"].append(score)
            else:
                grouped_entities.append(current_entity)
                current_entity = {"start": start, "end": end, "scores": [score]}

    if current_entity is not None:
        grouped_entities.append(current_entity)

    if not grouped_entities:
        print("Result: No mountain names detected in the text.")
    else:
        print("Mountain Entities Found:")
        print("-" * 40)
        for entity in grouped_entities:
            word = args.text[entity["start"]:entity["end"]]
            avg_score = sum(entity["scores"]) / len(entity["scores"])
            print(f" • {word} (Confidence: {avg_score:.1%})")


if __name__ == "__main__":
    inference_ner()