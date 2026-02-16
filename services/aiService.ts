
import { GoogleGenAI, Type } from "@google/genai";
import OpenAI from "openai";
import { Question, UserAnswer, GroundingLink } from "../types";

export interface AIExplanationResult {
  text: string;
  links?: GroundingLink[];
}

// --- Configuration & Helpers ---

const getGoogleKey = (): string => {
    // 1. Try standard process.env (Explicit access is required for many bundlers like Webpack/Vite to perform replacement)
    try {
        if (typeof process !== 'undefined' && process.env) {
            // @ts-ignore
            if (process.env.GOOGLE_API_KEY) return process.env.GOOGLE_API_KEY;
            // @ts-ignore
            if (process.env.API_KEY) return process.env.API_KEY;
            // @ts-ignore
            if (process.env.REACT_APP_GOOGLE_API_KEY) return process.env.REACT_APP_GOOGLE_API_KEY;
            // @ts-ignore
            if (process.env.VITE_GOOGLE_API_KEY) return process.env.VITE_GOOGLE_API_KEY;
        }
    } catch (e) {}

    // 2. Try import.meta.env (Vite/Modern ESM)
    try {
        // @ts-ignore
        if (typeof import.meta !== 'undefined' && import.meta.env) {
            // @ts-ignore
            if (import.meta.env.GOOGLE_API_KEY) return import.meta.env.GOOGLE_API_KEY;
            // @ts-ignore
            if (import.meta.env.API_KEY) return import.meta.env.API_KEY;
            // @ts-ignore
            if (import.meta.env.VITE_GOOGLE_API_KEY) return import.meta.env.VITE_GOOGLE_API_KEY;
        }
    } catch (e) {}

    return "";
};

const getOpenAIKey = (): string => {
    try {
        if (typeof process !== 'undefined' && process.env) {
            // @ts-ignore
            if (process.env.OPENAI_API_KEY) return process.env.OPENAI_API_KEY;
            // @ts-ignore
            if (process.env.REACT_APP_OPENAI_API_KEY) return process.env.REACT_APP_OPENAI_API_KEY;
            // @ts-ignore
            if (process.env.VITE_OPENAI_API_KEY) return process.env.VITE_OPENAI_API_KEY;
        }
    } catch (e) {}

    try {
        // @ts-ignore
        if (typeof import.meta !== 'undefined' && import.meta.env) {
            // @ts-ignore
            if (import.meta.env.OPENAI_API_KEY) return import.meta.env.OPENAI_API_KEY;
            // @ts-ignore
            if (import.meta.env.VITE_OPENAI_API_KEY) return import.meta.env.VITE_OPENAI_API_KEY;
        }
    } catch (e) {}

    return "";
};

// Determine active provider based on keys
const getActiveProvider = () => {
    if (getGoogleKey()) return 'GEMINI';
    if (getOpenAIKey()) return 'OPENAI';
    console.warn("AI Service: No valid API keys found. Checked GOOGLE_API_KEY, API_KEY, and OPENAI_API_KEY.");
    return null;
};

// Utility to clean markdown fences from JSON responses
const cleanJsonOutput = (text: string): string => {
  let clean = text.replace(/^```json\s*|\s*```$/g, '').replace(/^```\s*|\s*```$/g, '');
  const firstOpen = clean.indexOf('{');
  const lastClose = clean.lastIndexOf('}');
  if (firstOpen !== -1 && lastClose !== -1 && lastClose > firstOpen) {
    clean = clean.substring(firstOpen, lastClose + 1);
  }
  return clean;
};

// --- GEMINI DRIVER ---

const geminiDriver = {
    getClient: () => {
        const key = getGoogleKey();
        return key ? new GoogleGenAI({ apiKey: key }) : null;
    },

    getExplanation: async (question: Question, userAnswer: string | string[] | null, useWebSearch: boolean): Promise<AIExplanationResult> => {
        const ai = geminiDriver.getClient();
        if (!ai) throw new Error("Gemini Client Init Failed");

        const prompt = `
          You are an expert CISSP instructor and mentor. 
          A student is practicing for their exam and needs a deep conceptual explanation for the following question.
          
          QUESTION: ${question.question_text}
          CHOICES: ${JSON.stringify(question.choices)}
          CORRECT ANSWER: ${question.correct_answer}
          STUDENT ANSWER: ${userAnswer || "None provided"}
          DATABASE EXPLANATION: ${question.explanation}
          DOMAIN: ${question.domain} / ${question.subDomain}
          
          ${useWebSearch ? "CRITICAL: The student has requested a REAL-TIME WEB VERIFICATION. Use Google Search to verify if the technical details in this question/explanation match the LATEST 2024/2025 CISSP exam objectives and security standards. If the database info is outdated, point it out explicitly." : ""}

          Please provide:
          1. A simplified "In Plain English" summary of the core concept.
          2. Why the correct answer is right and why the specific student answer (if provided and incorrect) was wrong.
          3. A real-world scenario illustrating this principle.
          4. A "Memory Hook" or mnemonic to remember this concept.
          
          Keep the tone encouraging, professional, and concise. Format with clear headings and bullet points.
        `;

        const response = await ai.models.generateContent({
          model: 'gemini-3-pro-preview',
          contents: prompt,
          config: {
            maxOutputTokens: 6000,
            thinkingConfig: { thinkingBudget: 4000 },
            tools: useWebSearch ? [{ googleSearch: {} }] : []
          }
        });

        const text = response.text || "No explanation generated.";
        const links: GroundingLink[] = [];

        const groundingChunks = response.candidates?.[0]?.groundingMetadata?.groundingChunks;
        if (groundingChunks) {
          groundingChunks.forEach((chunk: any) => {
            if (chunk.web) links.push({ uri: chunk.web.uri, title: chunk.web.title });
          });
        }

        return { text, links: links.length > 0 ? links : undefined };
    },

    getStudyPlan: async (performanceData: any) => {
        const ai = geminiDriver.getClient();
        if (!ai) throw new Error("Gemini Client Init Failed");

        const prompt = `
          You are an expert CISSP Study Coach. Analyze the following quiz results and create a personalized study plan.
          RESULTS: ${JSON.stringify(performanceData)}
          
          Please provide a structured response in JSON format with the following keys:
          - "overallAssessment": A brief summary of the user's current readiness.
          - "topWeaknesses": An array of the top 3 domains or subdomains to focus on, with a brief reason why for each.
          - "studyStrategy": 3 actionable steps to improve before the next session.
          - "encouragement": A motivational closing sentence.
        `;

        const response = await ai.models.generateContent({
          model: "gemini-3-flash-preview",
          contents: prompt,
          config: {
            responseMimeType: "application/json",
            responseSchema: {
              type: Type.OBJECT,
              properties: {
                overallAssessment: { type: Type.STRING },
                topWeaknesses: {
                  type: Type.ARRAY,
                  items: {
                    type: Type.OBJECT,
                    properties: { topic: { type: Type.STRING }, reason: { type: Type.STRING } },
                    required: ["topic", "reason"]
                  }
                },
                studyStrategy: { type: Type.ARRAY, items: { type: Type.STRING } },
                encouragement: { type: Type.STRING }
              },
              required: ["overallAssessment", "topWeaknesses", "studyStrategy", "encouragement"]
            }
          }
        });

        return JSON.parse(cleanJsonOutput(response.text || "{}"));
    },

    getContextAnalysis: async (frictionPoints: any) => {
        const ai = geminiDriver.getClient();
        if (!ai) throw new Error("Gemini Client Init Failed");

        const prompt = `
          You are an elite CISSP diagnostician. Your task is to perform "Contextual Cross-Referencing" on a student's mistakes.
          Look beyond simple Domain labels and identify the semantic "connective tissue" or "friction points" that are causing errors.
          
          DATA: ${JSON.stringify(frictionPoints)}
          
          Provide a response with:
          1. "The Root Misconception": A clear description of the specific technical link or conceptual thread the student is misunderstanding.
          2. "Evidence": Briefly reference 2-3 specific questions from the data that prove this pattern exists.
          3. "The 'Mental Pivot'": A specific shift in thinking required to resolve this pattern.
          
          Format with clean, professional Markdown. Use bolding for emphasis. Keep it targeted and impactful.
        `;

        const response = await ai.models.generateContent({
          model: "gemini-3-pro-preview",
          contents: prompt,
          config: {
            maxOutputTokens: 8000,
            thinkingConfig: { thinkingBudget: 4000 }
          }
        });

        return response.text || "No analysis available.";
    }
};

// --- OPENAI DRIVER ---

const openAIDriver = {
    getClient: () => {
        const key = getOpenAIKey();
        return key ? new OpenAI({ apiKey: key, dangerouslyAllowBrowser: true }) : null;
    },

    getExplanation: async (question: Question, userAnswer: string | string[] | null, useWebSearch: boolean): Promise<AIExplanationResult> => {
        const client = openAIDriver.getClient();
        if (!client) throw new Error("OpenAI Client Init Failed");

        // OpenAI does not natively support Search Grounding in the base chat API without external tools.
        // We will notify the prompt about the inability to search if requested.
        const searchContext = useWebSearch 
            ? "NOTE: Real-time web search is unavailable in this mode. Rely on your training data (cutoff late 2023) to provide the most current standard interpretation available." 
            : "";

        const systemPrompt = `You are an expert CISSP instructor and mentor. Provide a deep conceptual explanation. ${searchContext}`;
        
        const userPrompt = `
            QUESTION: ${question.question_text}
            CHOICES: ${JSON.stringify(question.choices)}
            CORRECT ANSWER: ${question.correct_answer}
            STUDENT ANSWER: ${userAnswer || "None provided"}
            DATABASE EXPLANATION: ${question.explanation}
            DOMAIN: ${question.domain} / ${question.subDomain}

            Please provide:
            1. A simplified "In Plain English" summary of the core concept.
            2. Why the correct answer is right and why the student answer (if wrong) was wrong.
            3. A real-world scenario illustrating this principle.
            4. A "Memory Hook" or mnemonic.
            
            Keep the tone encouraging, professional, and concise. Format with clear Markdown headings and bullet points.
        `;

        const response = await client.chat.completions.create({
            model: "gpt-4o",
            messages: [
                { role: "system", content: systemPrompt },
                { role: "user", content: userPrompt }
            ]
        });

        return { 
            text: response.choices[0]?.message?.content || "No explanation generated.",
            links: undefined // OpenAI Standard API does not return source links
        };
    },

    getStudyPlan: async (performanceData: any) => {
        const client = openAIDriver.getClient();
        if (!client) throw new Error("OpenAI Client Init Failed");

        const prompt = `
          You are an expert CISSP Study Coach. Analyze the quiz results and create a personalized study plan.
          RESULTS: ${JSON.stringify(performanceData)}
          
          Respond in valid JSON format with:
          {
             "overallAssessment": "string",
             "topWeaknesses": [{"topic": "string", "reason": "string"}], // Top 3
             "studyStrategy": ["string", "string", "string"], // 3 steps
             "encouragement": "string"
          }
        `;

        const response = await client.chat.completions.create({
            model: "gpt-4o",
            response_format: { type: "json_object" },
            messages: [
                { role: "system", content: "You are a helpful study coach who outputs strict JSON." },
                { role: "user", content: prompt }
            ]
        });

        const raw = response.choices[0]?.message?.content || "{}";
        return JSON.parse(raw);
    },

    getContextAnalysis: async (frictionPoints: any) => {
        const client = openAIDriver.getClient();
        if (!client) throw new Error("OpenAI Client Init Failed");

        const prompt = `
          You are an elite CISSP diagnostician. Perform "Contextual Cross-Referencing" on a student's mistakes.
          Identify semantic "friction points".
          
          DATA: ${JSON.stringify(frictionPoints)}
          
          Provide:
          1. "The Root Misconception"
          2. "Evidence" (Reference 2-3 specific questions)
          3. "The 'Mental Pivot'": A specific shift in thinking required to resolve this pattern.
          
          Format with clean, professional Markdown. Bold for emphasis.
        `;

        const response = await client.chat.completions.create({
            model: "gpt-4o",
            messages: [
                { role: "system", content: "You are a precise technical analyst." },
                { role: "user", content: prompt }
            ]
        });

        return response.choices[0]?.message?.content || "No analysis available.";
    }
};

// --- PUBLIC API DISPATCHER ---

export const getAIExplanation = async (question: Question, userAnswer?: string | string[] | null, useWebSearch: boolean = false): Promise<AIExplanationResult> => {
    const provider = getActiveProvider();
    
    if (!provider) {
        return { text: "AI features are disabled. Please configure a valid Google Gemini or OpenAI API Key in your environment variables." };
    }

    try {
        if (provider === 'GEMINI') {
            return await geminiDriver.getExplanation(question, userAnswer || null, useWebSearch);
        } else {
            return await openAIDriver.getExplanation(question, userAnswer || null, useWebSearch);
        }
    } catch (error: any) {
        console.error(`${provider} AI Error:`, error);
        return { text: `Error generating explanation: ${error.message || "Unknown error"}` };
    }
};

export const getAIStudyPlan = async (answers: UserAnswer[]) => {
    const provider = getActiveProvider();

    if (!provider) {
        return {
            overallAssessment: "AI Study Plan Unavailable",
            topWeaknesses: [],
            studyStrategy: ["To enable AI analysis, please configure a valid API Key."],
            encouragement: "Keep studying!"
        };
    }

    try {
        const performanceData = answers.map(a => ({
            domain: a.question.domain,
            subDomain: a.question.subDomain,
            isCorrect: a.isCorrect,
            confidence: a.confidence,
            isSkipped: a.isSkipped
        }));

        if (provider === 'GEMINI') {
            return await geminiDriver.getStudyPlan(performanceData);
        } else {
            return await openAIDriver.getStudyPlan(performanceData);
        }
    } catch (error: any) {
        console.error(`${provider} Study Plan Error:`, error);
        return {
            overallAssessment: "Could not generate AI analysis due to an error.",
            topWeaknesses: [],
            studyStrategy: ["Please try again in a few moments."],
            encouragement: "Keep pushing forward!"
        };
    }
};

export const getAIContextualAnalysis = async (answers: UserAnswer[]) => {
    const provider = getActiveProvider();

    if (!provider) {
        return "Contextual Analysis is disabled. Please provide an API Key.";
    }

    try {
        const frictionPoints = answers.filter(a => !a.isCorrect || a.confidence !== 'high').map(a => ({
            text: a.question.question_text,
            explanation: a.question.explanation,
            domain: a.question.domain,
            subDomain: a.question.subDomain,
            topic: a.question.topic,
            isCorrect: a.isCorrect,
            confidence: a.confidence
        }));

        if (frictionPoints.length === 0) return "No friction points detected. You need to answer some questions incorrectly or with low confidence to generate an analysis.";

        if (provider === 'GEMINI') {
            return await geminiDriver.getContextAnalysis(frictionPoints);
        } else {
            return await openAIDriver.getContextAnalysis(frictionPoints);
        }
    } catch (error: any) {
        console.error(`${provider} Contextual Analysis Error:`, error);
        return `Error: ${error.message || "Analysis failed"}`;
    }
};
