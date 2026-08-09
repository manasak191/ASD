document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("upload-form");
  const resultDiv = document.getElementById("upload-result");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    resultDiv.innerHTML = "Analyzing...";

    const formData = new FormData(form);

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        resultDiv.innerHTML = `
          <div class="alert alert-success">
            <strong>File:</strong> ${data.filename}<br>
            <strong>Prediction:</strong> ${data.label}<br>
            <strong>Prob (Non_Autistic):</strong> ${(data.prob_healthy * 100).toFixed(1)}%<br>
            <strong>Prob (Autistic):</strong> ${(data.prob_patients * 100).toFixed(1)}%<br>
            <strong>Duration:</strong> ${data.duration_sec.toFixed(2)} sec
          </div>
        `;
      } else {
        resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
      }
    } catch (err) {
      resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${err.message}</div>`;
    }
  });
});
