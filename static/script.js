document.addEventListener("DOMContentLoaded", () => {
    let state = {
        file: null,
        isConverting: false,
        cancelTokenSource: null,
        isDarkMode: false,
        count: null,
        timer: null,
    };

    const elements = {
        fileInput: document.getElementById("fileInput"),
        convertBtn: document.getElementById("convertBtn"),
        cancelBtn: document.getElementById("cancelBtn"),
        darkModeToggle: document.getElementById("darkModeToggle"),
        timeRemaining: document.getElementById("timeRemaining"),
        countdown: document.getElementById("countdown"),
        audioContainer: document.getElementById("audioContainer"),
        audioPlayer: document.getElementById("audioPlayer"),
        downloadLink: document.getElementById("downloadLink"),
        urlToggle: document.getElementById("urlToggle"),
        fileToggle: document.getElementById("fileToggle"),
        urlInput: document.querySelector(".url-input"),
        fileInput: document.querySelector(".file-input"),
    };

    elements.urlToggle.addEventListener("click", () => {
        elements.urlToggle.classList.add("active");
        elements.fileToggle.classList.remove("active");
        elements.urlInput.style.display = "block";
        elements.fileInput.style.display = "none";
        elements.fileInput.value = "";
    });

    elements.fileToggle.addEventListener("click", () => {
        elements.fileToggle.classList.add("active");
        elements.urlToggle.classList.remove("active");
        elements.fileInput.style.display = "block";
        elements.urlInput.style.display = "none";
        elements.urlInput.value = "";
    });

    function formatTime(totalSeconds) {
        const hours = Math.floor(totalSeconds / 3600);
        const minutes = Math.floor((totalSeconds % 3600) / 60);
        const seconds = totalSeconds % 60;

        let timeString = [];
        if (hours > 0)
            timeString.push(`${hours} hour${hours !== 1 ? "s" : ""}`);
        if (minutes > 0)
            timeString.push(`${minutes} minute${minutes !== 1 ? "s" : ""}`);
        if (seconds > 0 || timeString.length === 0)
            timeString.push(`${seconds} second${seconds !== 1 ? "s" : ""}`);

        return timeString.join(", ");
    }

    function resetCountdown() {
        if (state.timer) {
            clearInterval(state.timer);
        }
        state.count = null;
        elements.timeRemaining.style.display = "none";
    }

    function updateUI(isConverting) {
        state.isConverting = isConverting;
        elements.timeRemaining.style.display = isConverting ? "inline" : "none";
        elements.cancelBtn.style.display = isConverting ? "inline" : "none";
        elements.convertBtn.disabled = isConverting;
    }

    elements.fileInput.addEventListener("change", (event) => {
        state.file = event.target.files[0];
        elements.audioContainer.style.display = "none";
    });

    elements.convertBtn.addEventListener("click", async () => {
        if (!state.file) return;

        const allowedExtensions = [".pdf", ".txt", ".docx"];
        if (
            !allowedExtensions.some((ext) =>
                state.file.name.toLowerCase().endsWith(ext),
            )
        ) {
            alert("Invalid file type! Please upload a PDF, TXT, or DOCX file.");
            return;
        }

        resetCountdown();
        updateUI(true);

        state.cancelTokenSource = axios.CancelToken.source();
        const formData = new FormData();
        formData.append("file", state.file);

        try {
            const countResponse = await axios.post("/count/", formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
            });

            state.count = parseInt(countResponse.data);
            state.timer = setInterval(() => {
                if (state.count > 0) {
                    state.count--;
                    elements.countdown.textContent = formatTime(state.count);
                } else {
                    clearInterval(state.timer);
                }
            }, 1000);

            const conversionResponse = await axios.post("/", formData, {
                headers: {
                    "Content-Type": "multipart/form-data",
                },
                cancelToken: state.cancelTokenSource.token,
            });

            elements.audioPlayer.src =
                "/audio/" + conversionResponse.data.audio_file;
            elements.downloadLink.href =
                "/downloads/" + conversionResponse.data.audio_file;
            elements.audioContainer.style.display = "block";
        } catch (error) {
            if (!axios.isCancel(error)) {
                const errorMessage =
                    error.response?.data?.error ||
                    "An unexpected error occurred during file conversion.";
                alert(errorMessage);
            }
        } finally {
            updateUI(false);
        }
    });

    elements.cancelBtn.addEventListener("click", () => {
        if (state.cancelTokenSource) {
            state.cancelTokenSource.cancel("Upload canceled by the user.");
            updateUI(false);
        }
    });

    elements.darkModeToggle.addEventListener("click", () => {
        state.isDarkMode = !state.isDarkMode;
        document.body.classList.toggle("dark-mode");
        elements.darkModeToggle.textContent = state.isDarkMode ? "🌙" : "☀️";
    });
});
