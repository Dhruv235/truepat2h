let speechQueue: string[] = [];
let isSpeaking = false;

export const speak = (text: string, interrupt: boolean = false) => {
  if (typeof window === 'undefined' || !window.speechSynthesis) return;

  if (interrupt) {
    window.speechSynthesis.cancel();
    speechQueue = [];
    isSpeaking = false;
  }

  speechQueue.push(text);
  processQueue();
};

const processQueue = () => {
  if (isSpeaking || speechQueue.length === 0) return;

  isSpeaking = true;
  const text = speechQueue.shift()!;
  const utterance = new SpeechSynthesisUtterance(text);

  utterance.rate = 1.0;
  utterance.pitch = 1.0;
  utterance.volume = 1.0;

  utterance.onend = () => {
    isSpeaking = false;
    processQueue();
  };

  utterance.onerror = () => {
    isSpeaking = false;
    processQueue();
  };

  window.speechSynthesis.speak(utterance);
};

export const stopSpeaking = () => {
  if (typeof window !== 'undefined' && window.speechSynthesis) {
    window.speechSynthesis.cancel();
    speechQueue = [];
    isSpeaking = false;
  }
};
