""" Tests for the Spec classes"""

from citoolkit.specifications.spec import UniverseSpec, NullSpec

###################################################################################################
# Basic Tests
###################################################################################################

def test_spec_universe_null():
    """ Tests that the Universe and Null Specs combine
    togther properly.
    """
    universe_spec = UniverseSpec()
    null_spec = NullSpec()

    assert (universe_spec & null_spec).explicit() == NullSpec()
    assert (universe_spec | null_spec).explicit() == UniverseSpec()
    assert (~universe_spec).explicit() == NullSpec()
    assert (~null_spec).explicit() == UniverseSpec()
