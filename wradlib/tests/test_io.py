
import wradlib.io as io

# testing functions related to readDX
def test___getTimestampFromFilename():
    pass

def test__getDXTimestamp():
    pass

def test__unpackDX():
    pass

def test__readDX():
    pass

def test__filter_gabella_a_tr1():
    data = np.arange(9)
    data[4] = 10
    result = cl._filter_gabella_a(data, tr1=5)
    assert result == 3

#-------------------------------------------------------------------------------
# testing the first part of the filter
#-------------------------------------------------------------------------------
def filter_setup():
    img = np.zeros((10,10), dtype=np.float32)
    img[2,2] = 10    # isolated pixel
    img[3,8:10] = 10 # line
    img[5,:] = 5     # spike
    img[7:9,7:9] = 5 # precip field

def test_filter_gabella_a():
    pass