import cv2
import sys
from ultralytics import YOLO

def main():
    if len(sys.argv) < 2:
        print("Usage: python manual_test_fight.py <0_for_webcam_or_video_path>")
        return
        
    path = sys.argv[1]
    
    print(f"Loading model: fight_yolov8.pt")
    model = YOLO("fight_yolov8.pt")
    
    # Check if path is an integer (webcam index)
    try:
        path = int(path)
    except ValueError:
        pass
    
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        print(f"Could not open {path} as video, trying as image...")
        img = cv2.imread(str(path))
        if img is None:
            print(f"Could not open {path} as image either.")
            return
            
        # Process single image
        results = model(img)
        res = results[0]
        probs = res.probs.data.cpu().numpy()
        print("\n--- Image Classification Results ---")
        for idx, name in res.names.items():
            print(f"[{idx}] {name}: {probs[idx]:.4f}")
            
        vi_idx = 1
        for idx, name in res.names.items():
             if "non" not in name.lower() and "viol" in name.lower():
                 vi_idx = idx
                 
        score = probs[vi_idx]
        color = (0, 0, 255) if score > 0.6 else (0, 255, 0)
        cv2.putText(img, f"Violence: {score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Manual Fight Model Test", img)
        cv2.waitKey(0)
    else:
        print(f"Playing video: {path}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
                
            results = model(frame, verbose=False)
            res = results[0]
            probs = res.probs.data.cpu().numpy()
            
            vi_idx = 1
            for idx, name in res.names.items():
                 if "non" not in name.lower() and "viol" in name.lower():
                     vi_idx = idx
                     
            score = probs[vi_idx]
            
            color = (0, 0, 255) if score > 0.6 else (0, 255, 0)
            
            # Draw score
            cv2.putText(frame, f"Violence Conf: {score:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            if score > 0.6:
                cv2.putText(frame, "CRITICAL: FIGHTING DETECTED", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            
            cv2.imshow("Manual Fight Model Test (Press q to exit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
