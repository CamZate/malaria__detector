let session;
const fileInput = document.getElementById("imageUpload");
const predictBtn = document.getElementById("predictBtn");
const resultEl = document.getElementById("result");
const preview = document.getElementById("preview");

async function initModel() {
  resultEl.textContent = "Loading model...";
  try {
    session = await ort.InferenceSession.create("malaria_model.onnx");
    resultEl.textContent = "Model loaded ✅";
  } catch (err) {
    console.error("Model load error:", err);
    resultEl.textContent = "❌ Failed to load model";
  }
}
initModel();

// Load image preview
fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = (ev) => {
      preview.src = ev.target.result;
    };
    reader.readAsDataURL(file);
  }
});

// Convert image to Float32 tensor (64x64x3)
async function preprocessImage(image) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  canvas.width = 64;
  canvas.height = 64;
  ctx.drawImage(image, 0, 0, 64, 64);
  const imgData = ctx.getImageData(0, 0, 64, 64);
  const { data } = imgData;

  // remove alpha and normalize
  const input = new Float32Array(64 * 64 * 3);
  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
    input[j] = data[i] / 255;     // R
    input[j + 1] = data[i + 1] / 255; // G
    input[j + 2] = data[i + 2] / 255; // B
  }

  // shape [1,64,64,3]
  return new ort.Tensor("float32", input, [1, 64, 64, 3]);
}

predictBtn.addEventListener("click", async () => {
  if (!fileInput.files.length) return alert("Please upload an image first!");
  resultEl.textContent = "Predicting... 🧠";

  const img = document.createElement("img");
  img.src = preview.src;
  await new Promise((res) => (img.onload = res));

  const inputTensor = await preprocessImage(img);

  const feeds = { input_layer_16: inputTensor };
  console.log("Running model with feeds:", feeds);

  try {
    const output = await session.run(feeds);
    console.log("Output:", output);

    const outputTensor = Object.values(output)[0].data[0];
    const prediction = outputTensor > 0.5 ? 1 : 0;
    const predictionLabel = prediction > 0.5 ? "✅ Uninfected" : "🦠 Infected";
    if(prediction === 1){
      resultEl.textContent = `Result: ${predictionLabel} (${(outputTensor * 100).toFixed(2)}% confidence)`;
    }else{
      resultEl.textContent = `Result: ${predictionLabel} (${((1-outputTensor) * 100).toFixed(2)}% confidence)`;
    }
  } catch (err) {
    console.error("Error during inference:", err);
    resultEl.textContent = "❌ Error running model";
  }
});
