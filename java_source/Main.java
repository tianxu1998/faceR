import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;

public class Main {
    public static void main(String[] args) {
//        args = new String[2];
//        args[0] = "http://localhost:8000/upload";
//        args[1] = "D:\\faceR\\djangoface\\Face\\face_dataset\\19001010105.JPG";
        System.out.println(uploadFile(args[0], "file", args[1], args.length >= 3 ? args[2] : null));
    }

    /**
     * ģ���ļ�post�ϴ�
     * @param urlStr���ӿڵ�ַ��
     * @param formName���ӿ�file��������
     * @param fileName����Ҫ�ϴ��ļ��ı���·����
     * @return�ļ��ϴ����ӿڷ��صĽ��
     */
    public static String uploadFile(String urlStr, String formName, String fileName, String... parm) {
        String name = fileName.substring(fileName.lastIndexOf(File.separator)+1);
        String baseResult = null;
        try {
            final String newLine = "\r\n";
            final String boundaryPrefix = "--";
            String BOUNDARY = "========7d4a6d158c9";// ģ�����ݷָ���
            URL url = new URL(urlStr);
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");// ����ΪPOST����
            conn.setDoOutput(true);
            conn.setDoInput(true);

            conn.setRequestProperty("connection", "Keep-Alive");// ��������ͷ����
            conn.setRequestProperty("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9");
            conn.setRequestProperty("Content-Type","multipart/form-data; boundary=" + BOUNDARY);
            conn.setRequestProperty("Accept-Encoding", "gzip, deflate, br");// ��������ͷ����
            conn.setRequestProperty("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8");

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

            out.write(sb.toString().getBytes());// ������ͷ������д�뵽�������
            InputStream in = new FileInputStream(file);// ����������,���ڶ�ȡ�ļ�����
            byte[] buf = new byte[1024];
            int len = 0;

            while ((len = in.read(buf)) != -1) {// ÿ�ζ�1KB����,���ҽ��ļ�����д�뵽�������
                out.write(buf, 0, len);
            }

            in.close();

            byte[] end_data = (boundaryPrefix + BOUNDARY
                    + boundaryPrefix + newLine).getBytes();

            out.write(end_data);
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
