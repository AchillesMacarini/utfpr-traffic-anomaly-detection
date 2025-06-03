import cv2

def background_subtraction_mog2(video_path, output_path=None, history=500, varThreshold=16, detectShadows=True):
    """
    Applies background subtraction using MOG2 to a video.

    Args:
        video_path (str): Path to the input video file.
        output_path (str, optional): Path to save the output video with foreground mask. If None, output is not saved.
        history (int): Length of the history.
        varThreshold (float): Threshold on the squared Mahalanobis distance to decide if it is well described by the background model.
        detectShadows (bool): Whether to detect shadows.

    Returns:
        None
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output video writer
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), False)

    # Create background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=detectShadows)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = backSub.apply(frame)

        if out:
            out.write(fg_mask)

    cap.release()
    if out:
        out.release()