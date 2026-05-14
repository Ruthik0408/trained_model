import React from "react";
/**
 * Error Boundary - Catches React component errors and displays fallback UI
 * Prevents entire app from crashing due to component errors
 */
export class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }
    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }
    componentDidCatch(error, errorInfo) {
        console.error("Error caught by boundary:", error, errorInfo);
    }
    render() {
        if (this.state.hasError) {
            return (this.props.fallback || (<div style={{
                    padding: "20px",
                    backgroundColor: "#fee",
                    color: "#c33",
                    borderRadius: "8px",
                    margin: "20px",
                }}>
            <h2>Something went wrong</h2>
            <p>{this.state.error?.message}</p>
            <button onClick={() => window.location.reload()}>Reload Page</button>
          </div>));
        }
        return this.props.children;
    }
}
