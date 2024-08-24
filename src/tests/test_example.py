import pytest


# This function will not be recognized as a test function by pytest
def helper_function():
    return "This is a helper function"


# pytest looks for files that start with test_ or end with _test, and within those files,
# it looks for functions that start with test_ or end with _test
def test_first():
    print("This is the first test")
    assert 1 == 1


# A fixture in pytest is a function that provides a fixed baseline or setup needed for tests.
# Fixtures can be used to set up state, provide dependencies, or perform cleanup after tests.
# They are defined using the @pytest.fixture decorator and can be used by simply adding them as arguments to test functions.

# The capsys fixture is a built-in pytest fixture that captures output to stdout and stderr.
# This is useful for testing functions that produce output to the console.
from .app import print_hello


# The capsys fixture is a built-in pytest fixture that captures output to stdout and stderr.
# This is useful for testing functions that produce output to the console.
# It captures the output of the print_hello function
def test_print_hello(capsys):
    print_hello()
    captured = capsys.readouterr()
    assert captured.out == "Hello, World!\n"


from .app import print_hello, multiply_by_two, divide_by_two


# The numbers fixture returns a list of two numbers, 10 and 20.
@pytest.fixture
def numbers():
    a = 10
    b = 20
    return [a, b]


class TestApp:
    def test_multiplication(self, numbers):
        res = multiply_by_two(numbers[0])
        assert res == numbers[1]

    def test_division(self, numbers):
        res = divide_by_two(numbers[1])
        assert res == numbers[0]

    def test_basic_multiplication(self):
        assert multiply_by_two(5) == 10

    # Parametrized tests allow you to run the same test with multiple different inputs.
    # This is useful for testing a function with a variety of inputs and expected outputs without writing separate test functions for each case.
    # You can use the @pytest.mark.parametrize decorator to specify the parameters and expected results for the test.
    @pytest.mark.parametrize(
        "input, expected", [(1, 2), (2, 4), (3, 6), (4, 8), (5, 10)]
    )
    def test_multiplication_param(self, input, expected):
        assert multiply_by_two(input) == expected

    @pytest.mark.parametrize(
        "input, expected", [(2, 1), (4, 2), (6, 3), (8, 4), (10, 5)]
    )
    def test_division_param(self, input, expected):
        assert divide_by_two(input) == expected


# Session-scoped fixture
@pytest.fixture(scope="session")
def session_fixture():
    print("\nSetup session fixture")
    return "session_fixture_value"


def test_using_session_fixture(session_fixture):
    print("Running test_using_session_fixture")
    assert session_fixture == "session_fixture_value"


def test_using_session_fixture_again(session_fixture):
    print("Running test_using_session_fixture_again")
    assert session_fixture == "session_fixture_value"


# Function-scoped fixture
@pytest.fixture(scope="function")
def function_fixture():
    print("\nSetup function fixture")
    return "function_fixture_value"


def test_using_function_fixture(function_fixture):
    print("Running test_using_function_fixture")
    assert function_fixture == "function_fixture_value"


def test_using_function_fixture_again(function_fixture):
    print("Running test_using_function_fixture")
    assert function_fixture == "function_fixture_value"
