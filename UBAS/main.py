class Sampler:
    """
    A template class for performing sampling of surrogate models
    """
    def __init__(self, name):
        """

        Parameters
        ----------
        name
        """
        self.name = name

    def get_name(self) -> str:
        """
        Returns
        -------
        str
        """
        return self.name
