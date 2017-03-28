\version "2.10.33"

\header {
  title = "220 - Sollt' ich meinem Gott nicht singen"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key d \minor
}


soprano = \relative c' {
  \repeat volta 2 {
    d4. e8 f8 g8 
    a2 d4 
    d4 e4 cis4 
    d2 d4 
    f2 e4     %5
    d2 c4 
    d8 c8 bes2 
    a2. }
  a2 a8 bes8 
  c2 f,4   %10
  bes2 a4 
  g2 g4 
  g2 a4 
  bes2 c4 
  bes4 a4 bes4   %15
  g2. 
  a2 a8 bes8 
  c2 a4 
  bes4 a2 
  g2.   %20
  f2 f4 
  f4 e4 f4 
  g4 f4 g4 
  a2 a4 
  a2 g4   %25
  a2 g4 
  a4 b2 
  c2. 
  d2 e4 
  f4. e8 d4   %30
  d4 e4 cis4 
  d2. 
  
}

alto = \relative c' {
  \repeat volta 2 {
    a4 d8 cis8 d8 e8 
    f8 e8 f8 g8 a4~ 
    a4 bes4 a4 
    a2 a4 
    a2 a4     %5
    f2 f4 
    f4 f8 g8 f8 e8 
    f2. }
  f2 f4~ 
  f8 g8 f8 e8 f4   %10
  f2 f4~ 
  f4 e8 d8 e4 
  e4 d4 d4 
  d4 g8 fis8 g4 
  g4 g4 fis4   %15
  d2. 
  f8 e8 f8 g8 d8 f8 
  g8 e8 f8 g8 a8 g8 
  f2 f4~ 
  f4 e2   %20
  f4 c2 
  d4 e4 d8 c8 
  d4 c8 d8 e4 
  f2 f4 
  f2 g4   %25
  g4 f4 e4 
  f2 g4 
  g2. 
  g4 a2 
  a4. g8 f4~   %30
  f4 e4 e8 g8 
  fis2. 
  
}

tenor = \relative c {
  \repeat volta 2 {
    f4. g8 a4~ 
    a4 d4. e8 
    f4 g4 e8 g8~ 
    g4 f8 e8 f4 
    d2 c4~     %5
    c4 bes4. a8 
    bes8 c8 d4 c4 
    c2. }
  d2 d4 
  c2 a4   %10
  d4 c2 
  c2 c4 
  c4 d4 a4~ 
  a4 g4 c8 d8 
  ees8 d8 ees8 c8 a8 d8   %15
  bes2. 
  d4 d4 d4 
  c2 c4 
  d4 c2 
  c2.   %20
  c4. bes8 a8 g8 
  a2 a4 
  g4 a4 bes4 
  c2 c4 
  c2 c4   %25
  c2 c4 
  c4 d2 
  e2. 
  bes4 f'4 e4~ 
  e4 d8 cis8 d8 c8   %30
  bes4 bes4 a4 
  a2. 
  
}

baixo = \relative c {
  \repeat volta 2 {
    d4 d,4 d'4 
    d8 cis8 d8 e8 f8 g8 
    a4 g4 a4 
    d,2 d,4 
    d'8 e8 f8 g8 a8 a,8     %5
    bes8 c8 d8 e8 f8 f,8 
    bes8 a8 g4 c4 
    f,2. }
  d'4 d'4 c8 bes8 
  a8 a,8 bes8 c8 d8 c8   %10
  d8 e8 f8 g8 a8 bes8 
  c2 c,4 
  c'4 bes8 a8 g8 fis8 
  g8 f8 ees8 d8 ees8 d8 
  c8 bes8 c4 d4   %15
  g,2. 
  d'8 cis8 d8 e8 f8 d8 
  e8 c8 d8 e8 f8 e8 
  d8 e8 f8 g8 a8 bes8 
  c2.   %20
  a8 bes8 a8 g8 f8 e8 
  d8 e8 d8 c8 bes8 a8 
  bes8 g8 c8 bes8 a8 g8 
  f2 f4 
  f8 a8 c8 f4 e8   %25
  f8 g8 a8 bes8 c8 c,8 
  f8 e8 d8 g8 f8 g8 
  c,2. 
  g8 g'8 f8 e8 d8 cis8 
  d8 f8 g8 a8 bes8 a8   %30
  g8 f8 g8 e8 a8 a,8 
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