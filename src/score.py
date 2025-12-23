import json
import torch
import sys
import os
import importlib.util


def init():
    global mil_model, device, model_dir
    
    model_dir = os.getenv("AZUREML_MODEL_DIR", "./models")
    
    print(f"🔍 Model directory: {model_dir}")
    print(f"📁 Files: {os.listdir(model_dir)}")
    
    potential_models_dir = os.path.join(model_dir, "models")
    if os.path.exists(potential_models_dir):
        model_dir = potential_models_dir
        print(f"📁 Found 'models' subdirectory, using: {model_dir}")
        print(f"📁 Files in models: {os.listdir(model_dir)}")
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    model_path = os.path.join(model_dir, "model.py")
    print(f"Loading model.py from: {model_path}")
    
    spec = importlib.util.spec_from_file_location("model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load model.py from {model_path}")
    
    model_module = importlib.util.module_from_spec(spec)
    sys.modules["model"] = model_module
    spec.loader.exec_module(model_module)
    
    AttentionMILClassifier = model_module.AttentionMILClassifier
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mil_path = os.path.join(model_dir, "model.pth")
    
    print(f"Loading model from {mil_path}...")
    
    mil_checkpoint = torch.load(mil_path, map_location=device, weights_only=False)
    
    # ✅ checkpoint에서 설정값 직접 가져오기
    model_config = mil_checkpoint['model_config']
    print(f"📊 Model config from checkpoint: {model_config}")
    
    # ✅ checkpoint의 설정값으로 모델 생성
    mil_model = AttentionMILClassifier(
        embedding_dim=model_config['embedding_dim'],
        num_classes=model_config['num_classes'],
        mil_hidden_dim=model_config['mil_hidden_dim'],
        use_gated_attention=model_config['use_gated_attention'],
        dropout=model_config['dropout']
    )
    
    mil_model.load_state_dict(mil_checkpoint['model_state_dict'])
    mil_model.to(device)
    mil_model.eval()
    
    print(f"✅ Model loaded on {device}")


def run(raw_data):
    try:
        data = json.loads(raw_data)
        print(f"📥 Received input data with embedding shape: {len(data['embedding'])}")
        
        embedding = torch.tensor(data["embedding"], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            embedding = embedding.unsqueeze(0).unsqueeze(0)
            print(f"🔄 Input shape after unsqueeze: {embedding.shape}")
            
            logits = mil_model(embedding)
            print(f"📊 Model output logits shape: {logits.shape}")
            
            predictions = torch.sigmoid(logits).cpu().numpy()
            print(f"✅ Predictions shape: {predictions.shape}")
            print(f"📈 Min/Max predictions: {predictions.min():.4f} / {predictions.max():.4f}")
        
        result = {
            "predictions": predictions[0].tolist(),
            "status": "success"
        }
        print(f"✔️  Inference completed successfully")
        return json.dumps(result)
    
    except Exception as e:
        error_msg = f"❌ Error in run: {str(e)}"
        print(error_msg)
        return json.dumps({"error": str(e), "status": "error"})
