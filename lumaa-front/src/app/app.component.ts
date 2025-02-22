import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, FormsModule, CommonModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css'
})
export class AppComponent {

  aws_url = "ec2-54-152-44-163.compute-1.amazonaws.com:8000/"
  local_url = "http://localhost:8000/"

  prompt: string = "";
  recommendations: any[] = [];
  showRecommendations: boolean = false;
  isLoading: boolean = false;

  constructor(private http: HttpClient) { }

  getRecommendations() {
    this.showRecommendations = false;
    this.isLoading = true;
    this.http.get(this.aws_url + "recommendation/" + this.prompt).subscribe((response: any) => {
      console.log(response)
      this.recommendations = response;
      this.showRecommendations = true;
      this.isLoading = false;
    });
  }

}
