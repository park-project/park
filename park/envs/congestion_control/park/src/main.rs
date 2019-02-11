use park::ParkAlg;

fn main() {
    println!("starting ccp shim alg");
    let addr = "/tmp/park-ccp".to_owned();
    portus::start!("netlink", None, ParkAlg(addr)).unwrap();
}
