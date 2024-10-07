import cv2
import numpy as np
import asyncio
from src.recognition import recognize_expression
from src.preprocessing import preprocess_image
from src.gemini_api import calculate_expression

async def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        preprocessed_frame = preprocess_image(frame)

        expression_task = asyncio.create_task(recognize_expression(preprocessed_frame))

        expression = await expression_task
        if expression:
            result_task = asyncio.create_task(calculate_expression(expression))
            result = await result_task
            print(f"Expression: {expression}, Result: {result}")

        cv2.imshow('Canvas', frame) #show

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())