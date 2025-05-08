const fs = require('fs');
const path = require('path');
require('dotenv').config(); // Load environment variables

(async () => {
    const fetch = (await import('node-fetch')).default;

    const HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large";
    const GROQ_API_URL = "https://api.groq.com/v1/chat/completions";

    const HUGGINGFACE_API_KEY = process.env.HUGGINGFACE_API_KEY;
    const GROQ_API_KEY = process.env.GROQ_API_KEY;

    if (!HUGGINGFACE_API_KEY || !GROQ_API_KEY) {
        console.error("❌ Missing API Keys in .env file");
        process.exit(1);
    }

    // Hugging Face Image Caption Function
    async function queryHuggingFace(filePath) {
        if (!fs.existsSync(filePath)) {
            throw new Error("Image file not found at path: " + filePath);
        }

        const data = fs.readFileSync(filePath);
        const response = await fetch(HUGGINGFACE_API_URL, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${HUGGINGFACE_API_KEY}`,
                "Content-Type": "application/octet-stream"
            },
            body: data
        });

        if (!response.ok) throw new Error(`Hugging Face API error: ${response.statusText}`);
        const result = await response.json();

        return result[0]?.generated_text || "No Caption Found";
    }

    // Groq Categorization Function
    async function categorizeWithGroq(text) {
        const prompt = `Given the description '${text}', classify the issue into one of the following categories:
"Cleanliness", "Staff Behavior", "Punctuality", "Water Availability", "Food Quality", "Security",
"Seating and Comfort", "Washroom Facilities", "Noise Disturbance", "Technical Malfunctions".

Return only the exact category name.`;

        const response = await fetch(GROQ_API_URL, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${GROQ_API_KEY}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                model: "llama3-8b-8192",
                messages: [{ role: "user", content: prompt }]
            })
        });

        if (!response.ok) throw new Error(`Groq API error: ${response.statusText}`);
        const result = await response.json();

        return result.choices?.[0]?.message?.content?.trim() || "Unknown Category";
    }

    // Main Execution
    try {
        const filePath = path.join(__dirname, 'hqdefault.jpg'); // Make sure image exists here
        const caption = await queryHuggingFace(filePath);
        console.log("\n📝 Hugging Face Caption:", caption);

        const category = await categorizeWithGroq(caption);
        console.log("📂 Groq Categorization:", category, "\n");
    } catch (error) {
        console.error("🚨 Error:", error.message);
    }
})();
