import cv2
import torch
import numpy as np

class BGRemover:

    def __init__(self, videoPath, outputRed, outputBG, GUI=False):
        self.capture = cv2.VideoCapture(videoPath)
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.progress_step = max(int(self.total_frames * 0.05), 1)
        self.last_progress = -1
        self.GUI = GUI
        if self.GUI:
            cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
            cv2.namedWindow('BG ESTATICO', cv2.WINDOW_NORMAL)
            cv2.namedWindow('OBJECTS IN RED', cv2.WINDOW_NORMAL)

        self.startOutput(outputRed, outputBG)
        self.checkIfCUDADevice()
        self.checkCapture()

    def checkIfCUDADevice(self):
        self.CUDA_enable = torch.cuda.is_available()
        print(f"CUDA disponível: {self.CUDA_enable}")

    def checkCapture(self):
        if not self.capture.isOpened():
            raise RuntimeError("O vídeo não foi carregado!!!!!")
        else:
            print("Vídeo foi carregado com sucesso")

    def startOutput(self, outputRed, outputBG):
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0  # fallback

        self.writerRed = cv2.VideoWriter(outputRed, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))
        self.writerBG = cv2.VideoWriter(outputBG, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))

    def startBGSubtract_CUDA(self):
        # Use torch for GPU frame processing, but background subtraction is still on CPU
        self.bgSubtractor = cv2.createBackgroundSubtractorMOG2()
        current_frame = 0
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break

            current_frame += 1
            progress = int((current_frame / self.total_frames) * 100)
            if progress % 5 == 0 and progress != self.last_progress:
                print(f"Progresso: {progress}%")
                self.last_progress = progress

            # Move frame to GPU using torch
            frame_tensor = torch.from_numpy(frame).cuda().float() / 255.0

            # If you want to do some GPU processing, do it here
            # For now, just move back to CPU for OpenCV processing
            frame_cpu = (frame_tensor * 255).byte().cpu().numpy()

            fg_mask = self.bgSubtractor.apply(frame_cpu, learningRate=0)
            background = self.bgSubtractor.getBackgroundImage()

            if background is not None:
                highlighted_view = background.copy()
                highlighted_view[fg_mask == 255] = (0, 0, 255)
                if self.GUI:
                    cv2.imshow('Original', frame)
                    cv2.imshow('BG ESTATICO', background)
                    cv2.imshow('OBJECTS IN RED', highlighted_view)

                self.writerRed.write(highlighted_view)
                self.writerBG.write(background)

            if self.GUI:
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

    def startBGSubtract(self):
        self.bgSubtractor = cv2.createBackgroundSubtractorMOG2()
        current_frame = 0
        while True:
            ret, frame = self.capture.read()
            if not ret:
                break

            current_frame += 1
            progress = int((current_frame / self.total_frames) * 100)
            if progress % 5 == 0 and progress != self.last_progress:
                print(f"Progresso: {progress}%")
                self.last_progress = progress

            fg_mask = self.bgSubtractor.apply(frame)
            background = self.bgSubtractor.getBackgroundImage()
            if background is not None:
                highlighted_view = background.copy()
                highlighted_view[fg_mask == 255] = (0, 0, 255)
                if self.GUI:
                    cv2.imshow('Original', frame)
                    cv2.imshow('BG ESTATICO', background)
                    cv2.imshow('OBJECTS IN RED', highlighted_view)

                self.writerRed.write(highlighted_view)
                self.writerBG.write(background)
                if self.GUI:
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break

    def BGSubtractor(self):
        print(self.CUDA_enable)
        if self.CUDA_enable:
            print("CUDA habilitado: usando GPU (via torch)")
            self.startBGSubtract_CUDA()
        else:
            print("CUDA não disponível: usando CPU")
            self.startBGSubtract()

        self.capture.release()
        self.writerRed.release()
        self.writerBG.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':

    path = 'D:/UTFPR/TCC/AI-City Challenge/aic21-track4-train-data/2.mp4'

    subtractor = BGRemover(path, 'red.mp4', 'BG.mp4', GUI=True)
    subtractor.BGSubtractor()
