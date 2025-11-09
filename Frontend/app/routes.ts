

import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
  index("routes/login.tsx"),
  route("/signup", "routes/signup.tsx"),
  route("/dashboard", "routes/dashboard.tsx"),
  route("/news", "routes/news.tsx"),
  route("/settings", "routes/settings.tsx"),
  route("/profile", "routes/profile.tsx"),
  route("/logout", "routes/logout.tsx"),
  route("*", "routes/notfound.tsx"),
] satisfies RouteConfig;