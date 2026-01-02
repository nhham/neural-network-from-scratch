var ocrDemo = {
    CANVAS_WIDTH: 200,
    TRANSLATED_WIDTH: 20,
    PIXEL_WIDTH: 10, // TRANSLATED_WIDTH = CANVAS_WIDTH / PIXEL_WIDTH
    BATCH_SIZE: 1, // Set to 1 for immediate feedback, or higher to batch
    HOST: "http://localhost",
    PORT: "8002",
    
    // Color configurations
    BLUE: "#e0eace",
    
    // State variables
    trainArray: [],
    trainingRequestCount: 0,
    data: null,

    onLoadFunction: function() {
        this.resetCanvas();
        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');

        // Bind events
        canvas.onmousemove = function(e) { ocrDemo.onMouseMove(e, ctx, canvas); };
        canvas.onmousedown = function(e) { ocrDemo.onMouseDown(e, ctx, canvas); };
        canvas.onmouseup = function(e) { ocrDemo.onMouseUp(e); };
    },

    resetCanvas: function() {
        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, this.CANVAS_WIDTH, this.CANVAS_WIDTH);
        
        // Initialize data array with 0s
        this.data = new Array(this.TRANSLATED_WIDTH * this.TRANSLATED_WIDTH).fill(0);
        
        this.drawGrid(ctx);
    },

    drawGrid: function(ctx) {
        for (var x = this.PIXEL_WIDTH, y = this.PIXEL_WIDTH; 
                 x < this.CANVAS_WIDTH; x += this.PIXEL_WIDTH, 
                 y += this.PIXEL_WIDTH) {
            ctx.strokeStyle = this.BLUE;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, this.CANVAS_WIDTH);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(this.CANVAS_WIDTH, y);
            ctx.stroke();
        }
    },

    onMouseMove: function(e, ctx, canvas) {
        if (!canvas.isDrawing) {
            return;
        }
        this.fillSquare(ctx, e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    },

    onMouseDown: function(e, ctx, canvas) {
        canvas.isDrawing = true;
        this.fillSquare(ctx, e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    },

    onMouseUp: function(e) {
        var canvas = document.getElementById('canvas');
        canvas.isDrawing = false;
    },

    fillSquare: function(ctx, x, y) {
        var xPixel = Math.floor(x / this.PIXEL_WIDTH);
        var yPixel = Math.floor(y / this.PIXEL_WIDTH);
        
        // Safety check to ensure we are inside canvas
        if(xPixel >=0 && xPixel < this.TRANSLATED_WIDTH && yPixel >= 0 && yPixel < this.TRANSLATED_WIDTH) {
             // Calculate array index. Note: The logic in the text used a slightly different indexing method.
             // This standardizes it: row-major order
             this.data[yPixel * this.TRANSLATED_WIDTH + xPixel] = 1;

            ctx.fillStyle = '#556b2f'; // Drawing color
            ctx.fillRect(xPixel * this.PIXEL_WIDTH, yPixel * this.PIXEL_WIDTH, 
                this.PIXEL_WIDTH, this.PIXEL_WIDTH);
        }
    },

    train: function() {
        var digitVal = document.getElementById("digit").value;
        if (!digitVal || this.data.indexOf(1) < 0) {
            alert("Please type and draw a digit value in order to train the network");
            return;
        }
        
        this.trainArray.push({"y0": this.data, "label": parseInt(digitVal)});
        this.trainingRequestCount++;

        if (this.trainingRequestCount == this.BATCH_SIZE) {
            alert("Sending training data to server...");
            var json = {
                trainArray: this.trainArray,
                train: true
            };

            this.sendData(json);
            this.trainingRequestCount = 0;
            this.trainArray = [];
        }
    },

    test: function() {
        if (this.data.indexOf(1) < 0) {
            alert("Please draw a digit in order to test the network");
            return;
        }
        var json = {
            image: this.data,
            predict: true
        };
        this.sendData(json);
    },

    receiveResponse: function(xmlHttp) {
        if (xmlHttp.status != 200) {
            alert("Server returned status " + xmlHttp.status);
            return;
        }
        var responseJSON = JSON.parse(xmlHttp.responseText);
        if (xmlHttp.responseText && responseJSON.type == "test") {
            var percent = Math.round(responseJSON.confidence * 100)

            alert("The neural network is " + percent + "% confident that you wrote a '" + responseJSON.result + "'");
        } else if (responseJSON.type == "train") {
            alert("Training complete!");
        }
    },

    onError: function(e) {
        alert("Error occurred while connecting to server: " + e.target.statusText);
    },

    sendData: function(json) {
        var xmlHttp = new XMLHttpRequest();
        xmlHttp.open('POST', this.HOST + ":" + this.PORT, false);
        xmlHttp.onload = function() { this.receiveResponse(xmlHttp); }.bind(this);
        xmlHttp.onerror = function() { this.onError(xmlHttp) }.bind(this);
        var msg = JSON.stringify(json);
        xmlHttp.setRequestHeader('Content-length', msg.length);
        xmlHttp.setRequestHeader("Connection", "close");
        xmlHttp.send(msg);
    }
};