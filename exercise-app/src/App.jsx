import ExerciseFormCorrector from "./components/ExerciseFormCorrector";
import ErrorBoundary from "./components/ErrorBoundary";

export default function App() {
  return (
    <ErrorBoundary>
      <ExerciseFormCorrector />
    </ErrorBoundary>
  );
}
