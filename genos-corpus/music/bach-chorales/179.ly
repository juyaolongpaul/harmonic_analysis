\version "2.10.33"

\header {
  title = "179 - Wachet auf, ruft uns die Stimme"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key ees \major
}


soprano = \relative c' {
  \repeat volta 2 {
    ees2 g2 
    bes2. bes4 
    bes2 bes2 
    c1 
    bes1     %5
    r2 bes2 
    ees2 bes2 
    ees4 f4 g2 
    f2 ees2 
    d2 c2     %10
    bes1 
    r2 bes2 
    ees2 bes2 
    c2 g2 
    aes4 g4 f2     %15
    ees1 }
  r2 bes'2 
  bes2. aes4 
  g2 f2 
  ees1   %20
  r2 bes'2 
  bes2. aes4 
  g2 f2 
  ees1 
  r2 f2   %25
  g2 aes2 
  g1 
  r2 bes2 
  c2 d2 
  ees2 ees4 f4   %30
  g2 f2 
  ees2 bes2 
  ees2 bes2 
  c2 g2 
  aes4 g4 f2   %35
  ees1 
  
}

alto = \relative c' {
  \repeat volta 2 {
    bes2 ees2 
    g2 f2 
    g2 f2 
    g2 f2 
    f1     %5
    r2 f2 
    bes2. aes4 
    g4 aes4 bes2 
    bes2 c2 
    c4 bes2 a4     %10
    f1 
    r2 g2 
    aes2 g2 
    ees2 ees2 
    ees2 d2     %15
    bes1 }
  r2 f'2 
  g2. f4 
  bes,4 c4 d2 
  c1   %20
  r2 d4 ees4 
  f2 f2 
  f4 ees2 d4 
  bes1 
  r2 d2   %25
  ees2. d4 
  ees1 
  r2 g2 
  g4 f4 f4 g4 
  g2 g4 aes4   %30
  bes2. aes4 
  g2 g4 aes4 
  bes4 aes4 bes4 g4 
  aes2 ees2 
  ees2 d2   %35
  bes1 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    g2 bes2 
    ees2 d2 
    ees2 bes2 
    bes2 a2 
    d1     %5
    r2 d2 
    ees2 ees2 
    bes2 ees2 
    f2 f2 
    f2. c4     %10
    d1 
    r2 ees2 
    ees2 ees4 des4 
    c2 c2 
    c4 bes4 aes2     %15
    g1 }
  r2 bes2 
  ees4 d4 c4 d4 
  ees2 bes4 aes4 
  g1   %20
  r2 bes2 
  bes4 c4 d2 
  bes4 c2 bes4 
  g1 
  r2 bes2   %25
  bes2 c4 aes4 
  bes1 
  r2 ees2 
  ees2 d2 
  c2 ees2   %30
  ees2 d2 
  ees2 ees2 
  bes2 ees2 
  ees2. c4 
  c4 bes4 aes2   %35
  g1 
  
}

baixo = \relative c {
  \repeat volta 2 {
    ees1~ 
    ees4 g4 bes4 aes4 
    g4 f4 ees4 d4 
    ees4 c4 f2 
    bes,1     %5
    r2 bes'4 aes4 
    g4 aes4 g4 f4 
    ees2 ees'2 
    d4 c4 bes4 a4 
    bes2 f2     %10
    bes,1 
    r2 ees4 des4 
    c4 d4 ees2 
    aes4 bes4 c2 
    f,2 bes,2     %15
    ees1 }
  r2 d2 
  ees2 f2 
  g4 aes4 bes4 bes,4 
  c1   %20
  r2 g'2 
  d4 c4 bes2 
  ees4 c4 aes4 bes4 
  ees1 
  r2 bes'4 aes4   %25
  g2 f2 
  ees1 
  r2 ees'2 
  a,2 b2 
  c2 c2   %30
  g4 aes4 bes2 
  c,2 ees4 f4 
  g4 f4 g4 ees4 
  aes,4 bes4 c2 
  f2 bes,2   %35
  ees1 
  
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