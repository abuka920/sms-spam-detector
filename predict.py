"""Command line spam check.

Usage:
    python predict.py "You've won a free cruise! Call now"
"""

import sys

from spam_model import load_model, predict


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    message = " ".join(sys.argv[1:])
    result = predict(load_model(), message)
    label = "SPAM" if result["is_spam"] else "HAM (legit)"
    print(f"{label}  |  spam probability: {result['spam_probability']:.1%}")
    for category, hits in result["red_flags"].items():
        print(f"  ⚠ {category}: {', '.join(hits)}")


if __name__ == "__main__":
    main()
