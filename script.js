let session;

async function initModel() {
  const status = document.getElementById("status");
  status.textContent = "Loading model...";
  session = await ort.InferenceSession.create("./malaria_model.onnx");
  status.textContent = "> Model ready <";
}

async function predict(image) {
  const status = document.getElementById("status");
  const predictionText = document.getElementById("prediction");
  const progressPath = document.getElementById("progress-path");
  const confidenceText = document.getElementById("confidence-text");

  status.textContent = "Scanning...";
  predictionText.textContent = "";
  progressPath.setAttribute("stroke-dasharray", "0,100");

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = 64;
  canvas.height = 64;
  ctx.drawImage(image, 0, 0, 64, 64);

  const imageData = ctx.getImageData(0, 0, 64, 64);
  const { data } = imageData;

  const input = new Float32Array(64 * 64 * 3);
  for (let i = 0; i < 64 * 64; i++) {
    input[i * 3 + 0] = data[i * 4 + 0] / 255;
    input[i * 3 + 1] = data[i * 4 + 1] / 255;
    input[i * 3 + 2] = data[i * 4 + 2] / 255;
  }

  const tensor = new ort.Tensor("float32", input, [1, 64, 64, 3]);
  const feeds = { input_layer_16: tensor };

  const results = await session.run(feeds);
  const output = results.output_0.data[0];
  const confidence = (output * 100).toFixed(2);

  confidenceText.textContent = `${confidence}%`;
  
  if (output < 0.5) {
    predictionText.innerHTML = `INFECTED<br><span style="font-size: 1rem; color: #6d6d6dff;">(${(100 - confidence).toFixed(2)}% confidence)</span>`;
    predictionText.style.color = "#ff4b4b";
    progressPath.setAttribute("stroke-dasharray", `${100-confidence},100`);
    progressPath.setAttribute("stroke", "#ff4b4b");
  } else {
    predictionText.innerHTML = `UNINFECTED<br><span style="font-size: 1rem; color: #6d6d6dff;">(${confidence}% confidence)</span>`;
    predictionText.style.color = "#50fa7b";
    progressPath.setAttribute("stroke-dasharray", `${confidence},100`);
    progressPath.setAttribute("stroke", "#50fa7b");
  }

  status.textContent = "Prediction complete.";
}

document.getElementById("fileInput").addEventListener("change", (e) => {
  const status = document.getElementById("status");
  const file = e.target.files[0];
  fileName = file ? file.name : "No file selected";
  document.getElementById('file-name').textContent = fileName;
  if (!file) return;
  const img = new Image();
  const reader = new FileReader();
  reader.onload = function (ev) {
    img.onload = function () {
      if (img.width > 400 || img.height > 400) {
        status.textContent = "Please upload cell image";
        e.target.value = "";
        return;
      }
      const canvas = document.getElementById("imageCanvas");
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const scale = Math.min(canvas.width / img.width, canvas.height / img.height);
      const x = (canvas.width - img.width * scale) / 2;
      const y = (canvas.height - img.height * scale) / 2;
      ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
      predict(img);
    };
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
});

initModel();
