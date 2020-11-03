import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SceneDatasetVideo(Dataset):
    def __init__(self, video_file, resizeTo = None):
        self.video_capture = cv2.VideoCapture(video_file)

        # cv2_im = cv2.cvtColor(cv_read_image, cv2.COLOR_BGR2RGB)

        if resizeTo is None:
            ret, cv_read_image = self.video_capture.read()
            image = Image.fromarray(cv_read_image)
            self.img_size = image.size

            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        else:
            self.img_size = resizeTo

            self.transform = transforms.Compose([
                transforms.Resize(resizeTo),
                transforms.ToTensor(),
            ])



    def __len__(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))-1

    def get_next_frame_number(self):
        return self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)

    def get_frame_size(self):
        return (int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    def __getitem__(self, idx):
        ret, cv_read_image = self.video_capture.read()
        # cv_read_image = cv2.cvtColor(cv_read_image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(cv_read_image)

        if self.transform:
            image = self.transform(image)

        return image, self.get_next_frame_number()-1



def get_dataset(video_file, batch_size=16, resizeTo=None):
    dataset_test = SceneDatasetVideo(video_file=video_file, resizeTo=resizeTo)
    return DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
