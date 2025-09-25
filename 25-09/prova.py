# scrivi una funzione slugyfi
def slugify(text):
    """Converts a string to a slugified version.

    Args:
        text (str): The input string to be slugified.

    Returns:
        str: The slugified string, with spaces replaced by hyphens and all characters in lowercase.
    """
    return text.lower().replace(" ", "-")

# sum of two numbers
def sum_of_two_numbers(a, b):
    return a + b


import unittest


class TestSlugify(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(slugify(""), "")

    def test_single_word(self):
        self.assertEqual(slugify("Hello"), "hello")

    def test_space_to_hyphen(self):
        self.assertEqual(slugify("Hello World"), "hello-world")

    def test_multiple_spaces(self):
        self.assertEqual(slugify("Multiple   Spaces"), "multiple---spaces")

    def test_existing_hyphen_preserved(self):
        self.assertEqual(slugify("Already-Slugified"), "already-slugified")

    def test_punctuation_preserved(self):
        self.assertEqual(slugify("Hello, World!"), "hello,-world!")

    def test_leading_trailing_spaces(self):
        self.assertEqual(slugify("  Hello  World  "), "--hello--world--")

    def test_only_spaces(self):
        self.assertEqual(slugify("   "), "---")

    def test_tabs_and_newlines_unchanged(self):
        self.assertEqual(slugify("\tHello\nWorld"), "\thello\nworld")

    def test_unicode_accents(self):
        self.assertEqual(slugify("Caffè latte"), "caffè-latte")

    def test_emojis_preserved(self):
        self.assertEqual(slugify("Hello 😊 World"), "hello-😊-world")

    def test_very_long_input(self):
        long_input = "A" * 10000
        self.assertEqual(slugify(long_input), "a" * 10000)

    def test_non_string_input_raises(self):
        with self.assertRaises(AttributeError):
            slugify(None)


if __name__ == "__main__":
    unittest.main()