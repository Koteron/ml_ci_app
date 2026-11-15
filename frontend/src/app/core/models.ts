export interface Chat {
  id: string;
  title: string;
  created_at: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  status: 'PENDING' | 'COMPLETED' | 'FAILED';
  created_at: string;
}

export interface GenerationParams {
  max_new_tokens: number;
  temperature: number;
  top_p: number;
}
