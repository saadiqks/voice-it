new Vue({
    el: "#app",
    data: {
        file: null,
        error: null,
        audioFile: null,
        isConverting: false,
        cancelTokenSource: null,
        isDarkMode: false,
    },
    // mounted() {
    //     if (localStorage.getItem("isDarkMode") === "true") {
    //         this.isDarkMode = true;
    //         document.body.classList.add("dark-mode");
    //     }
    // },
    methods: {
        handleFileUpload(event) {
            this.file = event.target.files[0];
            this.audioFile = null;
            this.error = null;
        },
        submitFile() {
            const allowedExtensions = [".pdf", ".txt", ".docx"];
            const fileName = this.file.name.toLowerCase();

            if (!allowedExtensions.some((ext) => fileName.endsWith(ext))) {
                alert(
                    "Invalid file type! Please upload a PDF, TXT, or DOCX file.",
                );
                return;
            }

            this.isConverting = true;
            this.error = null;
            this.audioFile = null;

            // Create cancel token
            this.cancelTokenSource = axios.CancelToken.source();

            let formData = new FormData();
            formData.append("file", this.file);

            axios
                .post("/", formData, {
                    headers: {
                        "Content-Type": "multipart/form-data",
                    },
                    cancelToken: this.cancelTokenSource.token, // Pass the cancel token
                })
                .then((response) => {
                    this.audioFile = response.data.audio_file;
                    this.error = null;
                })
                .catch((error) => {
                    if (axios.isCancel(error)) {
                        return;
                    } else if (
                        error.response &&
                        error.response.data &&
                        error.response.data.error
                    ) {
                        this.error = error.response.data.error;
                    } else {
                        this.error =
                            "An unexpected error occurred during file conversion.";
                    }
                    alert(this.error);
                })
                .finally(() => {
                    this.isConverting = false;
                });
        },
        cancelUpload() {
            if (this.cancelTokenSource) {
                this.cancelTokenSource.cancel("Upload canceled by the user.");
            }
        },
        toggleDarkMode(event) {
            this.isDarkMode = !this.isDarkMode;
            if (this.isDarkMode) {
                document.body.classList.add("dark-mode");
            } else {
                document.body.classList.remove("dark-mode");
            }
            // Save preference to localStorage
            // localStorage.setItem("isDarkMode", this.isDarkMode);
        },
    },
});
