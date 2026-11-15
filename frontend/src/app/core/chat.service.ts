import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { interval, switchMap, takeWhile, startWith } from 'rxjs';
import { Chat, Message, GenerationParams } from './models';

@Injectable({ providedIn: 'root' })
export class ChatService {
  private http = inject(HttpClient);
  private apiUrl = 'http://localhost:8000/chats';

  getChats() {
    return this.http.get<Chat[]>(this.apiUrl);
  }

  createChat(title = 'Новый чат') {
    return this.http.post<Chat>(this.apiUrl, { title });
  }

  deleteChat(chatId: string) {
    return this.http.delete(`${this.apiUrl}/${chatId}`);
  }

  getMessages(chatId: string) {
    return this.http.get<Message[]>(`${this.apiUrl}/${chatId}/messages`);
  }

  sendMessage(chatId: string, content: string, params: GenerationParams) {
    return this.http.post<Message>(`${this.apiUrl}/${chatId}/messages`, {
      content,
      params,
    });
  }

  pollMessagesUntilReady(chatId: string) {
    return interval(1500).pipe(
      startWith(0),
      switchMap(() => this.getMessages(chatId)),
      takeWhile((messages) => messages.some((m) => m.status === 'PENDING'), true)
    );
  }
}
