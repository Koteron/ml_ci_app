import { Component, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { AuthService } from '../../core/auth.service';

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [FormsModule],
  template: `
    <div class="auth-container">
      <div class="auth-card">
        <h2>{{ isLoginMode() ? 'Вход в систему' : 'Регистрация' }}</h2>
        <form (ngSubmit)="submit()">
          <input type="email" [(ngModel)]="email" name="email" placeholder="Email" required />
          <input type="password" [(ngModel)]="password" name="password" placeholder="Пароль (от 6 симв.)" required />
          @if (error()) {
            <div class="error">{{ error() }}</div>
          }
          <button type="submit">{{ isLoginMode() ? 'Войти' : 'Создать аккаунт' }}</button>
        </form>
        <button class="link-btn" (click)="isLoginMode.set(!isLoginMode())">
          {{ isLoginMode() ? 'Нет аккаунта? Зарегистрироваться' : 'Уже есть аккаунт? Войти' }}
        </button>
      </div>
    </div>
  `,
  styles: [`
    .auth-container { display: flex; height: 100vh; align-items: center; justify-content: center; background: #f3f4f6; font-family: sans-serif; }
    .auth-card { background: white; padding: 2rem; border-radius: 12px; width: 340px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    input { width: 100%; padding: 10px; margin-bottom: 12px; border: 1px solid #d1d5db; border-radius: 6px; box-sizing: border-box; }
    button[type="submit"] { width: 100%; padding: 10px; background: #2563eb; color: white; border: none; border-radius: 6px; cursor: pointer; font-weight: 600; }
    .link-btn { background: none; border: none; color: #2563eb; margin-top: 12px; cursor: pointer; width: 100%; }
    .error { color: #dc2626; font-size: 0.875rem; margin-bottom: 10px; }
  `]
})
export class LoginComponent {
  private auth = inject(AuthService);
  private router = inject(Router);

  isLoginMode = signal(true);
  email = '';
  password = '';
  error = signal('');

  submit() {
    this.error.set('');
    const req$ = this.isLoginMode()
      ? this.auth.login(this.email, this.password)
      : this.auth.register(this.email, this.password);

    req$.subscribe({
      next: () => this.router.navigate(['/']),
      error: (err) => this.error.set(err.error?.detail || 'Ошибка авторизации'),
    });
  }
}
