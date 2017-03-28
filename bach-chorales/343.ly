\version "2.10.33"

\header {
  title = "343 - Nun lieget alles unter dir"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key d \major
}


soprano = \relative c' {
  \repeat volta 2 {
    d4 d4 e4 
    fis4. gis8 a4 
    a2 gis4 
    a2 fis4 
    g2 fis4     %5
    e2 d4 
    e2. 
    d2. }
  e4 e4 fis4 
  g2 g4   %10
  fis4. e8 fis4 
  e2. 
  fis4 fis4 fis4 
  g4. a8 b4 
  b2 ais4   %15
  b2 cis4 
  d4 fis,2 
  g2 fis4 
  e4 e4 a4 
  b4. a8 g4   %20
  fis8 e8 e4. d8 
  d2. 
  
}

alto = \relative c' {
  \repeat volta 2 {
    a4 d4 d8 cis8 
    d2 cis8 a8 
    fis'4 e2 
    e2 d8 cis8 
    b8 cis8 d2~     %5
    d8 cis16 b16 cis4 d4~ 
    d4 cis8 b8 cis4 
    a2. }
  cis8 b8 cis4. dis8 
  e4 b4 e4~   %10
  e4 dis8 cis8 dis4 
  b2. 
  dis4 b4 dis4 
  e4 b4 b4 
  g'8 fis8 e8 d8 e4   %15
  d2 fis4 
  fis8 e8 d2 
  cis8 d8 e4 d4~ 
  d8 b8 cis4 d4 
  d4 g8 fis8 fis8 e8   %20
  d4~ d8 b8 cis4 
  a2. 
  
}

tenor = \relative c {
  \repeat volta 2 {
    fis4 fis8 d8 a'4 
    a8 fis8 b4 a4 
    b4 b8 cis8 d4 
    cis2 a4 
    g4 a8 g8 a4~     %5
    a4 a8 g8 fis4 
    b4 a4. g8 
    fis2. }
  a8 gis8 a4 a4
  b8 
  a8 g2    %10
  c4 b4. a8 
  g2. 
  b4 fis8 g8 a8 b8 
  g8 fis8 e4 e'8 d8 
  cis4. b8 cis4
  b2 ais4     %15
  b4. fis8 b8 a8 g8 
  fis8 e8 a4 gis8 
  a4 a4 a4 g8 
  a8 b8 cis8 d8 a8 a8   %20
  b8 a4 g4
  fis2. 
}

baixo = \relative c {
  \repeat volta 2 {
    d8 cis8 b4 a4 
    d4. e8 fis4 
    d8 b8 e2 
    a,2 d4 
    e4 fis4. g8     %5
    a8 g8 fis4 b8 a8 
    g8 e8 a4 a,4 
    d2. }
  a4 a'8 g8 fis4 
  e4. d8 c8 b8   %10
  a8 fis8 b4 b4 
  e2. 
  b8 cis8 dis4 b4 
  e4. fis8 g8 fis8 
  e8 d8 cis4 fis4   %15
  b,2 fis'4 
  b4. a8 g8 fis8 
  e8 d8 cis4 d4 
  a'4 a,4 fis'4 
  g4. a8 b8 cis,8   %20
  d8 g,8 a2 
  d2. 
  
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}