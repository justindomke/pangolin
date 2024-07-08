from cleanpangolin import util

def test_write_once_defalt_dict():
    d = util.WriteOnceDefaultDict(lambda p: 2*p)
    d[2] = 5
    assert d[2] == 5
    assert 2 in d
    assert 5 not in d
    assert d[5] == 10
    try:
        d[2] = 6
        assert False
    except ValueError:
        pass