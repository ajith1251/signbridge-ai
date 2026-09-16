"""
Text -> ISL finger-spelling (CLI).

Renders each A-Z character of the input using the generated local alphabet
assets (static/assets/isl_alphabet/, built by generate_alphabet_assets.py)
and displays the composite. All paths are resolved relative to the repository
root via isl_assets - no hardcoded drive paths.

Usage:
    python text_to_isl.py                # interactive prompt
    python text_to_isl.py "HELLO WORLD"  # convert the given text directly
"""

import string
import sys

import cv2
import numpy as np

from isl_assets import (
    TARGET_SIZE,
    assets_available,
    build_composite,
    load_letter_image,
    missing_assets,
)

ISL_LABELS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


def clean_input(text):
    allowed_chars = set(string.ascii_letters + string.digits + " ")
    cleaned = "".join(ch for ch in text if ch in allowed_chars)
    return cleaned.upper()


def text_to_isl_images(text):
    text = clean_input(text)
    images = []

    for char in text:
        if char == " ":
            # White spacer tile between words.
            images.append(np.ones((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8) * 255)
            continue
        if char not in ISL_LABELS:
            print(f"Skipping unsupported character: {char}")
            continue

        img = load_letter_image(char, TARGET_SIZE)
        if img is None:
            print(f"No reference asset for letter: {char}")
            continue
        images.append(img)

    return images


def display_isl_word(text):
    isl_images = text_to_isl_images(text)
    if not isl_images:
        print("No valid ISL images to display.")
        return

    combined_image = build_composite(isl_images)
    if combined_image is None:
        print("No valid ISL images to display.")
        return

    cv2.imshow("ISL Translation", combined_image)
    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    if not assets_available():
        print("Alphabet assets are missing:", ", ".join(missing_assets()))
        print("Generate them first with: python generate_alphabet_assets.py")
        sys.exit(1)

    if len(sys.argv) > 1:
        raw_input_text = " ".join(sys.argv[1:])
    else:
        raw_input_text = input("Enter a word or sentence to convert to ISL: ")

    user_input = clean_input(raw_input_text)
    if not any(ch.isalnum() for ch in user_input):
        print("No valid letters found.")
        sys.exit(1)

    display_isl_word(user_input)


if __name__ == "__main__":
    main()
