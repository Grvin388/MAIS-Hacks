import React, { useState, useRef } from "react";
import axios from "axios";

const ExerciseFormCorrector = () => {
  const [videoFile, setVideoFile] = useState(null);
  const [exerciseType, setExerciseType] = useState("squat");
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const videoRef = useRef(null);

  const exercises = [
    { value: "squat", label: "Squat" },
    { value: "pushup", label: "Push-up" },
    { value: "deadlift", label: "Deadlift" },
    { value: "lunge", label: "Lunge" },
    { value: "shoulder_press", label: "Shoulder Press" },
  ];

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setVideoFile(file);
      // Create preview URL
      const videoUrl = URL.createObjectURL(file);
      if (videoRef.current) {
        videoRef.current.src = videoUrl;
      }
    }
  };

  const analyzeForm = async () => {
    if (!videoFile) return;

    setLoading(true);
    setUploadProgress(0);

    const formData = new FormData();
    formData.append("video", videoFile);
    formData.append("exercise_type", exerciseType);

    try {
      const response = await axios.post(
        "http://localhost:5000/api/analyze",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setUploadProgress(percentCompleted);
          },
        }
      );

      setAnalysis(response.data);
    } catch (error) {
      console.error("Analysis failed:", error);
      alert("Analysis failed. Please try again.");
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };

  const getScoreColor = (score) => {
    if (score >= 80) return "text-green-600";
    if (score >= 60) return "text-yellow-600";
    return "text-red-600";
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case "critical":
        return "bg-red-100 border-red-400 text-red-700";
      case "warning":
        return "bg-yellow-100 border-yellow-400 text-yellow-700";
      case "info":
        return "bg-blue-100 border-blue-400 text-blue-700";
      default:
        return "bg-gray-100 border-gray-400 text-gray-700";
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            AI Exercise Form Corrector
          </h1>
          <p className="text-lg text-gray-600">
            Upload your exercise video and get instant feedback on your form
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          {/* Exercise Selection */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Exercise Type
            </label>
            <select
              value={exerciseType}
              onChange={(e) => setExerciseType(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {exercises.map((exercise) => (
                <option key={exercise.value} value={exercise.value}>
                  {exercise.label}
                </option>
              ))}
            </select>
          </div>

          {/* Video Upload */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Upload Exercise Video
            </label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center">
              <input
                type="file"
                accept="video/*"
                onChange={handleFileSelect}
                className="hidden"
                id="video-upload"
              />
              <label
                htmlFor="video-upload"
                className="cursor-pointer bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition duration-200"
              >
                Choose Video File
              </label>
              {videoFile && (
                <p className="mt-2 text-sm text-gray-600">
                  Selected: {videoFile.name}
                </p>
              )}
            </div>
          </div>

          {/* Video Preview */}
          {videoFile && (
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Video Preview
              </label>
              <video
                ref={videoRef}
                controls
                className="w-full max-w-md mx-auto rounded-lg shadow-md"
              >
                Your browser does not support the video tag.
              </video>
            </div>
          )}

          {/* Analyze Button */}
          <button
            onClick={analyzeForm}
            disabled={!videoFile || loading}
            className="w-full bg-green-600 text-white py-3 px-4 rounded-md hover:bg-green-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition duration-200 font-semibold"
          >
            {loading ? "Analyzing..." : "Analyze Exercise Form"}
          </button>

          {/* Upload Progress */}
          {loading && uploadProgress > 0 && (
            <div className="mt-4">
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
              <p className="text-sm text-gray-600 mt-1 text-center">
                Uploading: {uploadProgress}%
              </p>
            </div>
          )}
        </div>

        {/* Analysis Results */}
        {analysis && (
          <div className="bg-white rounded-lg shadow-md p-6 space-y-6">
            {/* Overall Score */}
            <div className="text-center border-b pb-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-2">
                Form Analysis Results
              </h2>
              <div className="inline-flex items-center justify-center w-24 h-24 bg-gray-100 rounded-full border-4 border-gray-300">
                <span
                  className={`text-2xl font-bold ${getScoreColor(
                    analysis.overall_score
                  )}`}
                >
                  {analysis.overall_score}%
                </span>
              </div>
              <p className="text-lg text-gray-600 mt-2">Overall Form Score</p>
            </div>

            {/* What's Right */}
            {analysis.whats_right && analysis.whats_right.length > 0 && (
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <h3 className="text-lg font-semibold text-green-800 mb-3 flex items-center">
                  <span className="w-3 h-3 bg-green-500 rounded-full mr-2"></span>
                  What You're Doing Right
                </h3>
                <ul className="space-y-2">
                  {analysis.whats_right.map((item, index) => (
                    <li key={index} className="text-green-700 flex items-start">
                      <span className="text-green-500 mr-2">✓</span>
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* Corrections Needed */}
            {analysis.corrections_needed &&
              analysis.corrections_needed.length > 0 && (
                <div className="space-y-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Corrections Needed
                  </h3>
                  <div className="space-y-3">
                    {analysis.corrections_needed.map((correction, index) => (
                      <div
                        key={index}
                        className={`border-l-4 p-4 rounded ${getSeverityColor(
                          correrection.severity
                        )}`}
                      >
                        <div className="flex justify-between items-start mb-1">
                          <h4 className="font-semibold">{correction.issue}</h4>
                          <span
                            className={`text-xs font-medium px-2 py-1 rounded-full ${
                              correction.severity === "critical"
                                ? "bg-red-200 text-red-800"
                                : correction.severity === "warning"
                                ? "bg-yellow-200 text-yellow-800"
                                : "bg-blue-200 text-blue-800"
                            }`}
                          >
                            {correction.severity}
                          </span>
                        </div>
                        <p className="text-sm mb-2">{correction.feedback}</p>
                        <div className="bg-white bg-opacity-50 p-3 rounded border">
                          <p className="text-sm font-semibold text-gray-700 mb-1">
                            How to correct:
                          </p>
                          <p className="text-sm">
                            {correction.correction_instruction}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

            {/* Detailed Breakdown */}
            {analysis.detailed_breakdown && (
              <div className="border-t pt-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">
                  Detailed Form Breakdown
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(analysis.detailed_breakdown).map(
                    ([aspect, data]) => (
                      <div key={aspect} className="bg-gray-50 rounded-lg p-4">
                        <div className="flex justify-between items-center mb-2">
                          <h4 className="font-medium text-gray-900 capitalize">
                            {aspect.replace("_", " ")}
                          </h4>
                          <span
                            className={`text-sm font-semibold ${getScoreColor(
                              data.score
                            )}`}
                          >
                            {data.score}%
                          </span>
                        </div>
                        <p className="text-sm text-gray-600">{data.feedback}</p>
                      </div>
                    )
                  )}
                </div>
              </div>
            )}

            {/* Improvement Tips */}
            {analysis.improvement_tips &&
              analysis.improvement_tips.length > 0 && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-blue-800 mb-3">
                    💡 Tips for Improvement
                  </h3>
                  <ul className="space-y-2">
                    {analysis.improvement_tips.map((tip, index) => (
                      <li key={index} className="text-blue-700">
                        {tip}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ExerciseFormCorrector;
