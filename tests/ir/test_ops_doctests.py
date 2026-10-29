from pangolin import ir
import doctest
import pytest
import inspect


def get_test_classes():
    classes = []
    for name, obj in inspect.getmembers(ir, inspect.isclass):
        if issubclass(obj, ir.ScalarOp) and obj is not ir.ScalarOp:
            classes.append(obj)
    return classes


@pytest.mark.parametrize("cls", get_test_classes())
def test_scalar_op(cls):

    finder = doctest.DocTestFinder()
    runner = doctest.DocTestRunner()

    tests = finder.find(cls)
    for test in tests:
        test.globs[cls.__name__] = cls  # Put the class in the test's namespace
        runner.run(test)

    if runner.failures > 0:
        pytest.fail(f"{cls.__name__} failed")


# test_dynamic_class(ir.Normal)
print(get_test_classes())
