import { Component, OnInit, OnDestroy, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { ChatService } from '../../core/chat.service';
import { AuthService } from '../../core/auth.service';
import { Chat, Message, GenerationParams } from '../../core/models';

@Component({
  selector: 'app-workspace',
  standalone: true,
  imports: [FormsModule],
  template: `
    <div class="layout">
      <!-- Сайдбар со списком чатов -->
      <aside class="sidebar">
        <button class="new-chat-btn" (click)="createNewChat()">+ Новый чат</button>
        <div class="chat-list">
          @for (chat of chats(); track chat.id) {
            <div
              class="chat-item"
              [class.active]="chat.id === activeChatId()"
              (click)="selectChat(chat.id)"
            >
              <span class="chat-title">{{ chat.title }}</span>
              <button class="del-btn" (click)="removeChat($event, chat.id)">×</button>
            </div>
          }
        </div>
        <button class="logout-btn" (click)="auth.logout()">Выйти</button>
      </aside>

      <!-- Основная область переписки -->
      <main class="chat-area">
        @if (activeChatId()) {
          <div class="messages">
            @for (msg of messages(); track msg.id) {
              <div class="message" [class.user]="msg.role === 'user'">
                <div class="role">{{ msg.role === 'user' ? 'Вы' : 'Wiki-NN' }}</div>
                <div class="bubble">
                  @if (msg.status === 'PENDING') {
                    <span class="typing">Генерация текста...</span>
                  } @else {
                    {{ msg.content }}
                  }
                </div>
              </div>
            }
          </div>

          <!-- Настройки и поле ввода -->
          <div class="composer">
            <div class="params-bar">
              <label>Длина: {{ params.max_new_tokens }}
                <input type="range" min="30" max="400" [(ngModel)]="params.max_new_tokens" />
              </label>
              <label>Температура: {{ params.temperature }}
                <input type="range" min="0.1" max="1.3" step="0.1" [(ngModel)]="params.temperature" />
              </label>
            </div>
            <div class="input-row">
              <textarea
                [(ngModel)]="promptText"
                placeholder="Введите начало текста или вопрос..."
                rows="2"
                (keydown.enter)="$event.preventDefault(); send()"
              ></textarea>
              <button [disabled]="isGenerating() || !promptText.trim()" (click)="send()">
                Отправить
              </button>
            </div>
          </div>
        } @else {
          <div class="empty-state">Создайте новый чат или выберите существующий слева</div>
        }
      </main>
    </div>
  `,
  styles: [`
    .layout { display: flex; height: 100vh; font-family: sans-serif; }
    .sidebar { width: 260px; background: #111827; color: white; display: flex; flex-direction: column; padding: 12px; }
    .new-chat-btn { padding: 10px; background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; margin-bottom: 12px; }
    .chat-list { flex: 1; overflow-y: auto; }
    .chat-item { display: flex; justify-content: space-between; align-items: center; padding: 10px; border-radius: 6px; cursor: pointer; margin-bottom: 4px; color: #d1d5db; }
    .chat-item:hover, .chat-item.active { background: #1f2937; color: white; }
    .chat-title { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 0.9rem; }
    .del-btn { background: none; border: none; color: #9ca3af; cursor: pointer; font-size: 1.1rem; }
    .logout-btn { padding: 8px; background: #374151; color: white; border: none; border-radius: 6px; cursor: pointer; }
    .chat-area { flex: 1; display: flex; flex-direction: column; background: #f9fafb; }
    .messages { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 16px; }
    .message { max-width: 75%; align-self: flex-start; }
    .message.user { align-self: flex-end; }
    .role { font-size: 0.75rem; color: #6b7280; margin-bottom: 4px; }
    .message.user .role { text-align: right; }
    .bubble { background: white; padding: 12px 16px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); line-height: 1.5; white-space: pre-wrap; }
    .message.user .bubble { background: #2563eb; color: white; }
    .typing { color: #6b7280; font-style: italic; }
    .composer { background: white; border-top: 1px solid #e5e7eb; padding: 12px 20px; }
    .params-bar { display: flex; gap: 20px; font-size: 0.85rem; color: #4b5563; margin-bottom: 8px; }
    .input-row { display: flex; gap: 10px; }
    textarea { flex: 1; padding: 10px; border: 1px solid #d1d5db; border-radius: 8px; resize: none; font-family: inherit; }
    .input-row button { padding: 0 20px; background: #2563eb; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; }
    .input-row button:disabled { background: #93c5fd; cursor: not-allowed; }
    .empty-state { margin: auto; color: #6b7280; }
  `]
})
export class WorkspaceComponent implements OnInit, OnDestroy {
  private chatService = inject(ChatService);
  auth = inject(AuthService);

  chats = signal<Chat[]>([]);
  activeChatId = signal<string | null>(null);
  messages = signal<Message[]>([]);
  isGenerating = signal(false);

  promptText = '';
  params: GenerationParams = { max_new_tokens: 120, temperature: 0.7, top_p: 0.9 };
  private pollSub?: Subscription;

  ngOnInit() {
    this.loadChats();
  }

  loadChats() {
    this.chatService.getChats().subscribe((list) => {
      this.chats.set(list);
      if (list.length > 0 && !this.activeChatId()) {
        this.selectChat(list[0].id);
      }
    });
  }

  createNewChat() {
    this.chatService.createChat().subscribe((chat) => {
      this.chats.update((prev) => [chat, ...prev]);
      this.selectChat(chat.id);
    });
  }

  selectChat(chatId: string) {
    this.pollSub?.unsubscribe();
    this.activeChatId.set(chatId);
    this.startPolling(chatId);
  }

  removeChat(event: Event, chatId: string) {
    event.stopPropagation();
    this.chatService.deleteChat(chatId).subscribe(() => {
      this.chats.update((prev) => prev.filter((c) => c.id !== chatId));
      if (this.activeChatId() === chatId) {
        this.activeChatId.set(null);
        this.messages.set([]);
      }
    });
  }

  send() {
    const chatId = this.activeChatId();
    const content = this.promptText.trim();
    if (!chatId || !content) return;

    this.promptText = '';
    this.isGenerating.set(true);

    this.chatService.sendMessage(chatId, content, this.params).subscribe({
      next: () => {
        this.loadChats(); // Обновляем заголовок первого сообщения
        this.startPolling(chatId);
      },
      error: () => this.isGenerating.set(false),
    });
  }

  private startPolling(chatId: string) {
    this.pollSub?.unsubscribe();
    this.pollSub = this.chatService.pollMessagesUntilReady(chatId).subscribe((msgs) => {
      this.messages.set(msgs);
      const pending = msgs.some((m) => m.status === 'PENDING');
      this.isGenerating.set(pending);
    });
  }

  ngOnDestroy() {
    this.pollSub?.unsubscribe();
  }
}
