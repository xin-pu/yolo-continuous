"""
Author: Xin.PU
Email: Pu.Xin@outlook.com
Time: 2022/9/6 9:44
"""


class TargetBox(object):
    top = 0
    left = 0
    bottom = 0
    right = 0
    score = 0
    label = ""
    color = None

    def __init__(self, box_xyxy, score, label, color):
        self.left = box_xyxy[0]
        self.top = box_xyxy[1]
        self.right = box_xyxy[2]
        self.bottom = box_xyxy[3]
        self.score = score
        self.label = label
        self.color = color

    def get_topleft(self):
        topleft = (self.left, self.top)
        return topleft

    def get_bottomright(self):
        bottomright = (self.right, self.bottom)
        return bottomright

    def __str__(self):
        info = "-" * 20 + type(self).__name__ + "-" * 20 + "\r\n"
        for key, value in self.__dict__.items():
            info += "%20s :\t%s\r\n" % (key, value)
        return info
