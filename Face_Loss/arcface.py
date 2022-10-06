from .arcmargin import ArcMarginHeader


class ArcFaceHeader(ArcMarginHeader):
    """ ArcFaceHeader class"""

    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super(ArcFaceHeader, self).__init__(in_features=in_features, out_features=out_features, s=s, m2=m)