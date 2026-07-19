class BoxInfo:
    def __init__(self, line):
        words = line.split()
        self.category = words.pop()
        words = [int(string) for string in words]
        self.player_ID = words[0]
        del words[0]

        x1, y1, x2, y2, frame_ID, lost, grouping, generated = words
        self.box = x1, y1, x2, y2
        self.frame_ID = frame_ID
        self.lost = lost
        self.grouping = grouping
        self.generated = generated


def load_tracking_annot(path):
    with open(path, 'r') as file:
        player_boxes = {idx: [] for idx in range(12)}
        frame_boxes_dct = {}

        for idx, line in enumerate(file):
            box_info = BoxInfo(line)
            if box_info.player_ID > 11:
                continue
            player_boxes[box_info.player_ID].append(box_info)

        for player_ID, boxes_info in player_boxes.items():
            boxes_info = boxes_info[5:]
            boxes_info = boxes_info[:-6]

            for box_info in boxes_info:
                if box_info.frame_ID not in frame_boxes_dct:
                    frame_boxes_dct[box_info.frame_ID] = []

                frame_boxes_dct[box_info.frame_ID].append(box_info)

        return frame_boxes_dct
