require("dotenv").config();
const axios = require("axios");

const API_KEY = process.env.YANDEX_API_KEY;
const FOLDER_ID = process.env.YANDEX_FOLDER_ID;

if (!API_KEY || !FOLDER_ID) {
    console.error("Нет YANDEX_API_KEY или YANDEX_FOLDER_ID в .env");
    process.exit(1);
}

const url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion";

const body = {
    modelUri: `gpt://${FOLDER_ID}/yandexgpt-lite`,
    completionOptions: { stream: false, temperature: 0.7, maxTokens: 120 },
    messages: [
        { role: "system", text: "Ты обычный зритель Twitch-стрима. Пиши коротко и по-человечески." },
        { role: "user", text: "Стример только что проиграл катку, напиши короткий комментарий в чат" }
    ],
};

axios
    .post(url, body, {
        headers: {
            Authorization: `Api-Key ${API_KEY}`,
            "Content-Type": "application/json",
        },
    })
    .then((r) => console.log("YandexGPT:", r.data.result.alternatives[0].message.text))
    .catch((e) => {
        console.error("Ошибка:", e.response?.status, e.response?.data || e.message);
    });
