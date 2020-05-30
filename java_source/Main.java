import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;

public class Main {
    public static void main(String[] args) {
        System.out.println(uploadFile(args[0], "file", args[1], args.length >= 3 ? args[2] : null));
    }

    /**
     * 模拟文件post上传
     * @param urlStr（接口地址）
     * @param formName（接口file接收名）
     * @param fileName（需要上传文件的本地路径）
     * @return文件上传到接口返回的结果
     */
    public static String uploadFile(String urlStr, String formName, String fileName, String... parm) {
        String name = fileName.substring(fileName.lastIndexOf(File.separator)+1);
        String baseResult = null;
        try {
            final String newLine = "\r\n";
            final String boundaryPrefix = "--";
            String BOUNDARY = "========7d4a6d158c9";// 模拟数据分隔线
            URL url = new URL(urlStr);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");// 设置为POST请求
            conn.setDoOutput(true);
            conn.setDoInput(true);

            conn.setRequestProperty("connection", "Keep-Alive");// 设置请求头参数
            conn.setRequestProperty("Charsert", "UTF-8");
            conn.setRequestProperty("Content-Type","multipart/form-data; boundary=" + BOUNDARY);
            OutputStream out = conn.getOutputStream();

            File file = new File(fileName);
            StringBuilder sb = new StringBuilder();
            sb.append(boundaryPrefix);
            sb.append(BOUNDARY);
            sb.append(newLine);
            if (parm != null) {
                sb.append("Content-Disposition: form-data;name=\""+"param" + "\"" + newLine);
                sb.append(newLine + parm[0] + newLine);
                sb.append(boundaryPrefix+BOUNDARY+newLine);
            }
            sb.append("Content-Disposition: form-data;name=\""+formName+"\";filename=\""+ name + "\"" + newLine);
            sb.append(newLine);
            sb.append(newLine);

            out.write(sb.toString().getBytes());// 将参数头的数据写入到输出流中
            FileInputStream in = new FileInputStream(file);// 数据输入流,用于读取文件数据
            byte[] buf = new byte[1024];
            int len = 0;

            while ((len = in.read(buf)) != -1) {// 每次读1KB数据,并且将文件数据写入到输出流中
                out.write(buf, 0, len);
            }

            out.write(newLine.getBytes());
            in.close();

            byte[] end_data = (newLine + boundaryPrefix + BOUNDARY
                    + boundaryPrefix + newLine).getBytes();

            out.write(end_data);
            out.flush();
            out.close();


            BufferedReader reader = new BufferedReader(new InputStreamReader(
                    conn.getInputStream()));
            String line = null;
            StringBuffer strs = new StringBuffer("");
            while ((line = reader.readLine()) != null) {
                strs.append(line);
            }
            baseResult = strs.toString();
        } catch (Exception e) {
            baseResult = e.getMessage();
        }
        return baseResult;
    }
}
