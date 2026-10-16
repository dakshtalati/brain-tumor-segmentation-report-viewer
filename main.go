package main

import (
	"encoding/json"
	"log"
	"net/http"
)

type HealthResponse struct {
	Status  string `json:"status"`
	Service string `json:"service"`
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	response := HealthResponse{
		Status:  "ok",
		Service: "brain-tumor-api",
	}

	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Println("Failed to write health check response", err)
	}
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", healthHandler)

	log.Println("Server running at https://localhost:8080")
	log.Fatal(http.ListenAndServe(":8080", mux))
}
