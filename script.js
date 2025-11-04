const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const predictBtn = document.getElementById("predictBtn");
const predictionText = document.getElementById("predictionText");

imageInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = function(event) {
      preview.src = event.target.result;
    };
    reader.readAsDataURL(file);
  }
});

predictBtn.addEventListener("click", async () => {
  if (!imageInput.files[0]) {
    alert("Please upload an image first!");
    return;
  }

  const formData = new FormData();
  formData.append("file", imageInput.files[0]);

  predictionText.innerText = "Analyzing...";

  try {
    const res = await fetch("https://malaria-backend-im7e.onrender.com/predict", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (data.error) throw new Error(data.error);

    predictionText.innerText = `Predicted: ${data.prediction} (${data.confidence}%)`;
  } catch (err) {
    predictionText.innerText = "Error: " + err.message;
  }
});

