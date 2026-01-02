from http.server import SimpleHTTPRequestHandler, HTTPServer
import json
from ocr import OCRNeuralNetwork

# Initialize the Neural Network
nn = OCRNeuralNetwork(15)

# This list will remember every drawing you send it
training_history = []

class OCRRequestHandler(SimpleHTTPRequestHandler):
    
    def do_POST(self):
        response_code = 200
        response = ""
        var_len = int(self.headers.get('Content-Length'))
        content = self.rfile.read(var_len)
        payload = json.loads(content.decode('utf-8'))

        if payload.get('train'):
            # Add the new drawing to our history
            new_data = payload['trainArray']
            training_history.extend(new_data)
            
            # Train on EVERYTHING and the whole history we have seen so far 
            # This prevents it from forgetting the old numbers.
            print(f"Training on {len(training_history)} samples...")
            for _ in range(50):
                nn.train(training_history)
            
            nn.save()
            response = {"type": "train", "status": "success"}
        
        elif payload.get('predict'):
            try:
                img_data = payload['image']
                prediction, confidence = nn.predict(img_data)
                response = {
                    "type": "test", 
                    "result": prediction,
                    "confidence": confidence
                }
            except Exception as e:
                print(f"Error: {e}")
                response_code = 500
        else:
            response_code = 400

        self.send_response(response_code)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if response:
            self.wfile.write(json.dumps(response).encode('utf-8'))

def run(server_class=HTTPServer, handler_class=OCRRequestHandler, port=8002):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting OCR Server on port {port}...')
    print(f'Open your browser to: http://localhost:{port}/ocr.html')
    httpd.serve_forever()

if __name__ == '__main__':
    run()

